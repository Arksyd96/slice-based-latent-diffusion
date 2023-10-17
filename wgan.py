import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path 
import json
from modules.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock


class WGAN(pl.LightningModule):
    def __init__(
        self,
        in_channels     = 3, 
        out_channels    = 3, 
        spatial_dims    = 2,
        emb_channels    = 4,
        hid_chs         = [32, 64, 128, 256],
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        lr              = 1e-4,
        lr_scheduler    = None,
        dropout         = 0.0,
        use_res_block   = True,
        learnable_interpolation = True,
        use_attention   = 'none',
        **kwargs
    ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        NORM = 'batch'
        ACT = 'relu'
        # NORM = ("GROUP", {'num_groups': 32, "affine": True})
        # ACT = ("Swish", {})


        ####################################################
        # ----------------- Discriminator ------------------
        ####################################################

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=ACT, 
            norm_name=NORM,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = NORM,
                act_name = ACT,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = None
            ) for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 3),
            BasicBlock(spatial_dims, emb_channels, emb_channels, 1)
        )

        self.discriminator = nn.Sequential(
            self.inc,
            *self.encoders,
            self.out_enc
        )  

        ####################################################
        # ----------------- Generator ----------------------
        ####################################################

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=ACT, norm_name=NORM, emb_channels=None) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=NORM,  
                act_name=ACT, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth - 1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)

        self.generator = nn.Sequential(
            self.inc_dec,
            *self.decoders[::-1],
            self.outc
        )
    
    def encode(self, x, emb=None):
        h = self.inc(x, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z, emb=None):
        h = self.inc_dec(z, emb=emb)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h, emb=emb)
        x = self.outc(h)
        return x 

    def forward(self, x_in, timestep=None):
        # --------- Time Embedding -----------
        emb = self.encode_timestep(timestep)

        # --------- Encoder --------------
        h = self.inc(x_in, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q, emb=emb)
        for i in range(len(self.decoders) - 1, -1, -1):
            if i < len(self.outc_ver):
                out_hor.append(self.outc_ver[i](h))  
            h = self.decoders[i](h, emb=emb)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss 

    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'
        
        # compute reconstruction loss
        perceptual_loss = self.perception_loss(pred[:, 0, None], target[:, 0, None]) if self.use_perceptual_loss else 0
        ssim_loss = self.ssim_loss(pred, target) if self.use_ssim_loss else 0
        pixel_loss = self.loss_fct(pred, target)

        loss = torch.mean(perceptual_loss + ssim_loss + pixel_loss)

        return loss
    
    def _step(self, batch, batch_idx, split, step):
        # ------------------------- Get Source/Target ---------------------------
        # x, t = batch
        x, = batch
        target = x
        
        if self.time_embedder is None:
            t = None
        
        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x, timestep=t)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss * self.embedding_loss_weight
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss': loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['mask_rec_loss'] = torch.sum(self.loss_fct(pred, target)) / pred.shape[0]      

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log('{}/{}'.format(split, metric_name), metric_val, prog_bar=True,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
    
        return loss
    
    def configure_optimizers(self):
        opt_vqvae = self.optimizer_vqvae(self.vqvae.parameters(), **self.optimizer_vqvae_kwargs)
        opt_gan = self.optimizer_gan(self.discriminator.parameters(), **self.optimizer_gan_kwargs)
        schedulers = []
        if self.lr_scheduler_vqvae is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_vqvae(opt_vqvae, **self.lr_scheduler_vqvae_kwargs), 
                'interval': 'step',
                'frequency': 1
            })
        if self.lr_scheduler_gan is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_gan(opt_gan, **self.lr_scheduler_gan_kwargs),
                'interval': 'step',
                'frequency': 1
            })
        return [opt_vqvae, opt_gan], schedulers
    
    def compute_lambda(self, rec_loss, gan_loss, dec_out_layer, eps=1e-4):
        """Computes adaptive weight as proposed in eq. 7 of https://arxiv.org/abs/2012.09841"""
        rec_grads = torch.autograd.grad(rec_loss, dec_out_layer.weight, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, dec_out_layer.weight, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps) 
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        return d_weight.detach()
    
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