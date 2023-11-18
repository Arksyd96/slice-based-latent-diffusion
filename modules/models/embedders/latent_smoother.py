import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from modules.models.utils.conv_blocks import UnetResBlock

class LatentSmoother(pl.LightningModule):
    def __init__(self, channels, latent_embedder=None):
        super().__init__()
        self.channels = channels

        # --- architecture ---
        self.encoder = UnetResBlock(2, channels, channels, kernel_size=3, norm_name='batch', blocks=3)
        self.decoder = UnetResBlock(2, channels, channels, kernel_size=3, norm_name='batch', blocks=3)
        self.latent_embedder = latent_embedder

        # --- loss ---
        self.loss = nn.MSELoss()

        self.save_hyperparameters(ignore=['latent_embedder'])

    def encode_as_2d_latent(self, x: torch.Tensor, return_individual_latents=False):
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

    def encode(self, z):
        z_c = self.encoder(z)
        return z_c
    
    def decode(self, z_c):
        z = self.decoder(z_c)
        return z

    def forward(self, z):
        z_c = self.encode(z)
        z = self.decode(z_c)
        return z, z_c
    
    def training_step(self, batch, batch_idx):
        x, = batch # => [Bx2x128x128x64]
        z = self.encode_as_2d_latent(x) # => [Bx6x128x128]
        z_hat, _ = self.forward(z) # => [Bx6x128x128]
        loss = self.loss(z_hat, z)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, = batch
        z = self.encode_as_2d_latent(x)
        z_hat, _ = self(z)
        loss = self.loss(z_hat, z)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer