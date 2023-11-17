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

    def encode_as_2d_latent(self, x: torch.Tensor):
        # probably giving it in a batch format
        if self.latent_embedder is not None:
        # Embed into latent space or normalize 
            batch = []
            self.latent_embedder.eval() 
            with torch.no_grad():
                for idx in range(x.shape[0]): # for each patient [2x128x128x64]
                    volume = x[idx].permute(3, 0, 1, 2) # => [64x2x128x128]
                    latents = self.latent_embedder.encode(volume, emb=None) # => [64x6x16x16]
                    batch.append(latents)

            x = torch.stack(batch, dim=0) # => [Bx64x6x16x16]
            x = x.permute(0, 2, 3, 4, 1) # => [Bx6x16x16x64]
            
            self.orig_latent_shape = x.shape[2:] # [16x16x64]
            w = h = torch.sqrt(torch.prod(self.orig_latent_shape)).long() # sqrt(16x16x64) = 128
            x = x.reshape(x.shape[0], self.channels, w, h) # => [Bx6x128x128]

        return x

    def decode_as_3d_volume(self, x: torch.Tensor):
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

    def encode(self, x):
        x = self.encode_as_2d_latent(x)
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
    
    def decode_as_volume(self, z):
        x_hat = self.decode(z)
        x_hat = self.decode_as_3d_volume(x_hat)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z = self(x)
        loss = self.loss(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z = self(x)
        loss = self.loss(x_hat, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer