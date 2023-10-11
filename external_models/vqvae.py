import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path 
import json

from modules.loss.perceivers import LPIPS
from modules.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from pytorch_msssim import SSIM, ssim

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_channels, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_channels = emb_channels
        self.beta = beta
  
        self.embedder = nn.Embedding(num_embeddings, emb_channels)
        self.embedder.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        assert z.shape[1] == self.emb_channels, "Channels of z and codebook don't match"
        z_ch = torch.moveaxis(z, 1, -1) # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels) # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (    torch.sum(z_flattened ** 2, dim=1, keepdim=True) 
                 +  torch.sum(self.embedder.weight ** 2, dim=1)
                -2* torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t())
        ) # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1) # [Bx*]
        print(min_encoding_indices, min_encoding_indices.unique())
        z_q = self.embedder(min_encoding_indices) # [Bx*, C]
        z_q = z_q.view(z_ch.shape) # [Bx*, C] -> [B, *, C] 
        z_q = torch.moveaxis(z_q, -1, 1) # [B, *, C] -> [B, C, *]
 
        # Compute Embedding Loss 
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
     
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss
    
    def sample(self, batch_size, device):
        indices = torch.randint(0, self.num_embeddings, (batch_size,), device=device)
        return self.embedder(indices)
    
class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels     = 3, 
        out_channels    = 3, 
        spatial_dims    = 2,
        emb_channels    = 4,
        num_embeddings  = 8192,
        hid_chs         = [32, 64, 128, 256],
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        lr              = 1e-4,
        lr_scheduler    = None,
        dropout         = 0.0,
        use_res_block   = True,
        learnable_interpolation = True,
        use_attention   = 'none',
        embedding_loss_weight = 1.0,
        beta            = 0.25,
        # perceiver = LPIPS, 
        # perceiver_kwargs = {},
        # perceptual_loss_weight = 1.0,
        **kwargs
    ):
        super().__init__()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        norm_name = ("GROUP", {'num_groups': 32, "affine": True})
        act_name = ("Swish", {})
        self.embedding_loss_weight = embedding_loss_weight
        # self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        # self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides) 
        self.kwargs = kwargs
        self.depth = len(hid_chs)

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock 
        self.inc = ConvBlock(
            spatial_dims, in_channels, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0],
            act_name=act_name, norm_name=norm_name
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims, 
                hid_chs[i - 1], 
                hid_chs[i], 
                kernel_sizes[i], 
                strides[i],
                kernel_sizes[i], 
                norm_name,
                act_name,
                dropout,
                use_res_block,
                learnable_interpolation,
                use_attention[i]
            ) for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 1)

        # ----------- Quantizer --------------
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, 
            emb_channels=emb_channels,
            beta=beta
        )    

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name)

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims, 
                hid_chs[i + 1], 
                hid_chs[i],
                kernel_size=kernel_sizes[i + 1], 
                stride=strides[i + 1], 
                upsample_kernel_size=strides[i + 1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                skip_channels=0
            ) for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        return z 
            
    def decode(self, z):
        z, _ = self.quantizer(z)
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i - 1](h)
        x = self.outc(h)
        return x 
    
    def decode_post_quantization(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i - 1](h)
        x = self.outc(h)
        return x
    
    def sample(self, batch_size, device):
        z_q = self.quantizer.sample(batch_size, device)
        return self.decode_post_quantization(z_q)

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            h = self.decoders[i](h)
        out = self.outc(h)
   
        return out, emb_loss 
    
    # def perception_loss(self, pred, target):
    #     if (self.perceiver is not None):
    #         self.perceiver.eval()
    #         return self.perceiver(pred, target)*self.perceptual_loss_weight
    #     else:
    #         return 0 

    # def ssim_loss(self, pred, target):
    #     return 1 - ssim(
    #         ((pred + 1) / 2).clamp(0, 1), 
    #         (target.type(pred.dtype) + 1) / 2, 
    #         data_range = 1, 
    #         size_average=False, 
    #         nonnegative_ssim=True
    #     ).reshape(-1, *[1] * (pred.ndim - 1))

    def rec_loss(self, pred, target):
        pixel_loss = F.l1_loss(pred, target, reduction='none')
        # perceptual_loss = self.perception_loss(pred, target)
        # ssim_loss = self.ssim_loss(pred, target)
        loss = torch.mean(pixel_loss)
        return loss 

    def _step(self, batch, batch_idx, state):
        # ------------------------- Get Source/Target ---------------------------
        x, = batch
        target = x

        # ------------------------- Run Model ---------------------------
        pred, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, target)
        loss += emb_loss * self.embedding_loss_weight

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss':loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim((pred + 1) / 2, (target.type(pred.dtype) + 1) / 2, data_range = 1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     
    
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.kwargs.get('optimizer_kwargs', {}))
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.kwargs.get('lr_scheduler_kwargs', {}))
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
        
    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, version=0, **kwargs):
        path_version = 'lightning_logs/version_'+str(version)
        with open(Path(path_checkpoint_dir) / path_version/ 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir, version)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 