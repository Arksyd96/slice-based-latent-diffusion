import numpy as np
import os
import torch
import pytorch_lightning as pl
import wandb
from torchvision.utils import save_image

class ImageReconstructionLogger(pl.Callback):
    def __init__(
        self, 
        n_samples = 5,
        sample_every_n_epochs = 1,
        save      = True, 
        save_dir  = os.path.curdir, 
        **kwargs
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.save = save
        self.sample_every_n_epochs = sample_every_n_epochs
        self.save_dir = '{}/images'.format(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # sampling only when master node master process
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.sample_every_n_epochs == 0: 
            pl_module.eval()
            
            with torch.no_grad():    
                batch = trainer.val_dataloaders.dataset.sample(self.n_samples)
                x, pos = x.to(pl_module.device, torch.float32), pos.to(pl_module.device, torch.long)
                
                if pl_module.time_embedding is None:
                    pos = None
                
                x_hat, _, _ = pl_module(x, timestep=pos)
                
                # at this point x and x_hat are of shape [B, 2, 128, 128]
                originals = torch.cat([
                    torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
                ], dim=0)
                
                reconstructed = torch.cat([
                    torch.hstack([img for img in x_hat[:, idx, ...]]) for idx in range(x_hat.shape[1])
                ], dim=0)
                
                img = torch.cat([originals, reconstructed], dim=0)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        img.detach().cpu().numpy(), 
                        caption='{} - {} (Top are originals)'.format(self.modalities, trainer.current_epoch)
                    )
                })
                
                if self.save:
                    x, x_hat = x.reshape(-1, 1, *x.shape[2:]), x_hat.reshape(-1, 1, *x_hat.shape[2:])
                    images = torch.cat([x, x_hat], dim=0)
                    save_image(
                        images, '{}/sample_{}.png'.format(self.save_dir, trainer.current_epoch), 
                        nrow=x.shape[0], normalize=True
                    )