import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from modules.models.utils.conv_blocks import UnetResBlock
from modules.models.embedders.latent_embedders import VAE

class LatentSmoother(VAE):
    def __init__(
        self,
        channels,
        latent_embedder,
        spatial_dims = 2,
        hid_chs = [64, 128, 256, 512],
        kernel_sizes = [3, 3, 3, 3],
        strides = [1, 2, 2, 2],
        dropout = None,
        use_res_block = True,
        learnable_interpolation = True,
        use_attention = 'none',
        embedding_loss_weight = 1e-6,
        optimizer = torch.optim.Adam, 
        optimizer_kwargs = {'lr':1e-4},
        lr_scheduler = None, 
        lr_scheduler_kwargs = {},
        loss = torch.nn.L1Loss,
        loss_kwargs = {'reduction': 'none'}
    ):
        super().__init__(
            channels, channels, spatial_dims, channels, hid_chs, kernel_sizes, strides,
            dropout = dropout,
            use_res_block = use_res_block,
            learnable_interpolation = learnable_interpolation,
            use_attention = use_attention,
            embedding_loss_weight = embedding_loss_weight,
            optimizer = optimizer, 
            optimizer_kwargs = optimizer_kwargs,
            lr_scheduler = lr_scheduler, 
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            loss = loss,
            loss_kwargs = loss_kwargs,
            use_ssim_loss = False,
            use_perceptual_loss = False
        )

        self.channels = channels
        self.latent_embedder = latent_embedder
        self.latent_embedder.freeze()
        self.save_hyperparameters(ignore=['latent_embedder'])

    def encode_as_2d_latent(self, x: torch.Tensor, return_individual_latents=True):
        if self.latent_embedder is not None:
            # Embed into latent space or normalize 
            batch = []
            self.latent_embedder.eval() 
            with torch.no_grad():
                for idx in range(x.shape[0]): # for each patient [2x128x128x64]
                    volume = x[idx].permute(3, 0, 1, 2) # => [64x2x128x128]
                    latents = self.latent_embedder.encode(volume, emb=None) # => [64x6x16x16]
                    batch.append(latents)

            z_i = torch.stack(batch, dim=0) # => [Bx64x6x16x16]
            z_i = z_i.permute(0, 2, 3, 4, 1) # => [Bx6x16x16x64]
            
            self.orig_latent_shape = z_i.shape[2:] # [16x16x64]
            w = h = np.sqrt(np.prod(self.orig_latent_shape)).astype(np.uint64) # sqrt(16x16x64) = 128
            Z = z_i.reshape(z_i.shape[0], self.channels, w, h) # => [Bx6x128x128]

            if return_individual_latents:
                return Z, z_i
            return Z

    def decode_as_volume(self, x: torch.Tensor):
        assert self.orig_latent_shape is not None, "You need to encode the latent space first"
        
        if self.latent_embedder is not None:
            x = x.reshape(x.shape[0], self.channels, *self.orig_latent_shape) # => [Bx6x16x16x64]
            x = x.permute(0, 4, 1, 2, 3) # => [Bx64x6x16x16]

            # Embed into latent space or normalize 
            batch = []
            self.latent_embedder.eval() 
            with torch.no_grad():
                for idx in range(x.shape[0]): # => [64x6x16x16]
                    volume = self.latent_embedder.decode(x[idx]) # => [64x2x128x128]
                    volume = volume.permute(1, 2, 3, 0) # => [2x128x128x64]
                    batch.append(volume)

            x = torch.stack(batch, dim=0) # => [Bx64x2x128x128]
        
        return x
    
    def step(self, batch, batch_idx, split):
        x, = batch # => [Bx2x128x128x64]
        z = self.encode_as_2d_latent(x, return_individual_latents=False) # => [Bx6x128x128]
        z_hat, z_hor, kl_loss = self.forward(z) # => [Bx6x128x128]
        
        # computing loss -------------
        pixel_loss = self.rec_loss(z_hat, z_hor, z) # => [Bx6x128x128]
        loss = pixel_loss + kl_loss * self.embedding_loss_weight

        # logging --------------------
        self.log('{}_loss'.format(split), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_pixel_loss'.format(split), pixel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_kl_loss'.format(split), kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer