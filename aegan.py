import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path 
import json
import os
import wandb
from torchvision.utils import save_image
from modules.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
import gc

class AutoEncodingWGAN(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        in_channels     = 3, 
        out_channels    = 3, 
        spatial_dims    = 2,
        emb_channels    = 4,
        hid_chs         = [32, 64, 128, 256],
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        lr              = 0.0002,
        lr_scheduler    = None,
        dropout         = 0.0,
        use_res_block   = True,
        learnable_interpolation = True,
        use_attention   = 'none',
        g_iter          = 1,
        d_iter          = 1,
        **kwargs
    ):
        super().__init__()
        self.input_shape = np.array(input_shape)
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.latent_shape = (emb_channels, *self.input_shape // (2 ** (self.depth - 1)))
        self.lr = lr
        self.lr_scheduler = lr_scheduler   
        self.g_iter = g_iter
        self.d_iter = d_iter
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        NORM = 'batch'
        ACT = 'relu'
        
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
        # ----------------- Code Disc ----------------------
        ####################################################
        self.code_discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.latent_shape), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

        ####################################################
        # ----------------- Encoder ------------------------
        ####################################################
        self.enc_inc = ConvBlock(
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
        self.enc_encoders = nn.ModuleList([
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
        self.enc_out = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 3),
            BasicBlock(spatial_dims, emb_channels, emb_channels, 1)
        )

        self.encoder = nn.Sequential(
            self.enc_inc,
            *self.enc_encoders,
            self.enc_out
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
            self.outc,
            nn.Tanh()
        )

        # -------------------------------------------------------

        self.save_hyperparameters()
        self.automatic_optimization = False

    def set_trainable(self, g, e, d, cd):
        for p in self.generator.parameters():
            p.requires_grad = g
        for p in self.encoder.parameters():
            p.requires_grad = e
        for p in self.discriminator.parameters():
            p.requires_grad = d
        for p in self.code_discriminator.parameters():
            p.requires_grad = cd
    
    def training_step(self, batch, batch_idx, split='train'):
        g_optimizer, e_optimizer, d_optimizer, cd_optimizer = self.optimizers()
        # ------------------------- Get Source/Target ---------------------------
        real_x, = batch

        ####################################################
        # --------- Train Generator + Encoder --------------
        ####################################################
        # generator, encoder, discriminator, code_discriminator in order
        self.set_trainable(True, True, False, False)

        for iter in range(self.g_iter):
            g_optimizer.zero_grad(set_to_none=True)
            e_optimizer.zero_grad(set_to_none=True)
            
            z = torch.randn(real_x.shape[0], *self.latent_shape).to(real_x.device, non_blocking=True)

            z_hat = self.encoder(real_x)
            fake_x = self.generator(z_hat).detach()
            fake_x_rand = self.generator(z).detach()
            c_loss = -self.code_discriminator(z_hat).mean()

            d_real_loss = self.discriminator(fake_x).mean()
            d_fake_loss = self.discriminator(fake_x_rand).mean()
            d_loss = -(d_real_loss + d_fake_loss)
            l1_loss = 10 * torch.nn.functional.l1_loss(fake_x, real_x)
            loss = d_loss + l1_loss + c_loss

            if iter < self.g_iter - 1:
                self.manual_backward(loss)
            else :
                self.manual_backward(loss, retain_graph=True)
            g_optimizer.step()
            e_optimizer.step()

        ####################################################
        # ----------------- Train Discriminator ------------
        ####################################################
        self.set_trainable(False, False, True, False)
        
        for iter in range(self.d_iter):
            d_optimizer.zero_grad(set_to_none=True)
            # z = torch.randn(real_x.shape[0], *self.latent_shape).to(real_x.device, non_blocking=True)

            z_hat = self.encoder(real_x).detach()
            fake_x = self.generator(z_hat).detach()
            fake_x_rand = self.generator(z).detach()
            x_loss = -2 * self.discriminator(real_x).mean() + self.discriminator(fake_x).mean() + self.discriminator(fake_x_rand).mean()
            gradient_penalty_r = self.calc_gradient_penalty(self.discriminator, real_x, fake_x_rand)
            gradient_penalty_f = self.calc_gradient_penalty(self.discriminator, real_x, fake_x)

            loss_2 = x_loss + gradient_penalty_r + gradient_penalty_f
            self.manual_backward(loss_2, retain_graph=True)
            d_optimizer.step()

        ####################################################
        # ----------------- Train CD -----------------------
        ####################################################
        self.set_trainable(False, False, False, True)

        for iter in range(self.d_iter):
            cd_optimizer.zero_grad(set_to_none=True)
            # z = torch.randn(real_x.shape[0], *self.latent_shape).to(real_x.device, non_blocking=True)

            gradient_penalty_cd = self.calc_gradient_penalty(self.code_discriminator, z_hat, z)
            loss_3 = -self.code_discriminator(z).mean() - c_loss + gradient_penalty_cd

            self.manual_backward(loss_3, retain_graph=True)
            cd_optimizer.step()

        gc.collect()

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {
                'c_loss': c_loss, 'g_loss': loss, 'd_loss': loss_2, 'cd_loss': loss_3,
            }

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log('{}/{}'.format(split, metric_name), metric_val, prog_bar=True,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        e_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        cd_opt = torch.optim.Adam(self.code_discriminator.parameters(), lr=self.lr)

        return g_opt, e_opt, d_opt, cd_opt
    
    def sample(self, n_samples=1, device='cuda'):
        z = torch.randn(n_samples, *self.latent_shape).to(device)
        return self.generator(z)
    
    def calc_gradient_penalty(self, model, x, x_gen, w=10):
        """WGAN-GP gradient penalty"""
        _eps = 1e-15

        assert x.size() == x_gen.size(), "real and sampled sizes do not match"
        alpha_size = tuple((len(x), *(1,) * (x.dim() - 1)))
        alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
        alpha = alpha_t(*alpha_size).uniform_()
        x_hat = x.data * alpha + x_gen.data * (1 - alpha)
        x_hat = x_hat.requires_grad_(True)

        def eps_norm(x):
            x = x.view(len(x), -1)
            return (x * x + _eps).sum(-1).sqrt()
        def bi_penalty(x):
            return (x-1) ** 2

        grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

        penalty = w * bi_penalty(eps_norm(grad_xhat)).mean()
        return penalty
    
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
    
class AEGANGenerator(pl.Callback):
    def __init__(self, sample_every_n_epochs = 5, save = True, save_dir = os.path.curdir) -> None:
        super().__init__()
        self.save = save
        self.sample_every_n_epochs = sample_every_n_epochs
        self.save_dir = '{}/images'.format(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.sample_every_n_epochs == 0:
            with torch.no_grad():
                generated = pl_module.sample(n_samples=1).detach().squeeze(0)
                # => 2, 192, 192, 96

                generated = generated.permute(3, 0, 1, 2)
                generated = generated[::4, ...] # 96 // 6 = 16

                if self.save:
                    save_image(generated[:, 0, None], '{}/sample_images_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)

                generated = torch.cat([
                    torch.hstack([img for img in generated[:, idx, ...]]) for idx in range(generated.shape[1])
                ], dim=0)

                generated = generated.add(1).div(2).mul(255)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        generated.cpu().numpy(), 
                        caption='[{}]'.format(trainer.current_epoch)#, format_condition(condition[0].cpu().numpy()))
                    )
                })