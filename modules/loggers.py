from typing import Any
import numpy as np
import os
import torch
import pytorch_lightning as pl
import wandb
from torchvision.utils import save_image
from modules.funcs import reverse_spatial_stack

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
                for dataset, split in zip(
                    [trainer.train_dataloader.dataset, trainer.val_dataloaders.dataset], 
                    ['train', 'val']
                ):
                    batch = dataset.sample(self.n_samples)
                    # x, pos = batch
                    # x, pos = x.to(pl_module.device, torch.float32), pos.to(pl_module.device, torch.long)
                    
                    x = batch[0]
                    x = x.to(pl_module.device, torch.float32)

                    if pl_module.time_embedder is None:
                        pos = None
                    
                    x_hat, _, _ = pl_module(x, timestep=pos)
                    
                    # if not self.is_3d:
                    # at this point x and x_hat are of shape [B, 2, 128, 128]
                    originals = torch.cat([
                        torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
                    ], dim=0)
                    
                    reconstructed = torch.cat([
                        torch.hstack([img for img in x_hat[:, idx, ...]]) for idx in range(x_hat.shape[1])
                    ], dim=0)
                    
                    img = torch.cat([originals, reconstructed], dim=0)
                    # else:
                    #     # shape [B, 2, 128, 128, 64]
                    #     x = x[:, :, :, :, ::4]
                    #     x = x[:, 0].permute(0, 3, 2, 1)
                        
                    #     x_hat = x_hat[:, :, :, :, ::4]
                    #     x_hat = x_hat[:, 0].permute(0, 3, 2, 1) # i suppose slices are channels
                        
                    #     originals = torch.cat([
                    #         torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
                    #     ], dim=0)
                        
                    #     reconstructed = torch.cat([
                    #         torch.hstack([img for img in x_hat[:, idx, ...]]) for idx in range(x_hat.shape[1])
                    #     ], dim=0)
                        
                    #     img = torch.cat([originals, reconstructed], dim=0)

                    # [-1, 1] => [0, 255]
                    img = img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                    
                    wandb.log({
                        'Reconstruction examples': wandb.Image(
                            img.detach().cpu().numpy(), 
                            caption='{} - {} (Top are originals)'.format(split, trainer.current_epoch)
                        )
                    })
                    
                    if self.save:
                        x, x_hat = x.reshape(-1, 1, *x.shape[2:]), x_hat.reshape(-1, 1, *x_hat.shape[2:])
                        images = torch.cat([x, x_hat], dim=0)
                        save_image(images, '{}/sample_{}_{}.png'.format(self.save_dir, split, trainer.current_epoch), nrow=x.shape[0],
                                   normalize=True)
                        

class ImageGenerationLogger(pl.Callback):
    def __init__(
        self,
        noise_shape,
        save_every_n_epochs = 5,
        save      = True,
        save_dir  = os.path.curdir
    ) -> None:
        super().__init__()
        self.save = save
        self.save_every_n_epochs = save_every_n_epochs
        self.noise_shape = noise_shape
        self.save_dir = '{}/images'.format(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            pl_module.eval()

            with torch.no_grad():
                condition = trainer.train_dataloader.dataset.sample(1) # 1x64x2x128x128
                condition = condition[0][:, :, 1, None, ...] # 1x64x1x128x128
                print(condition.shape)
                condition = pl_module.condition_latent_embedder.encode(
                    condition.squeeze(0).to(pl_module.condition_latent_embedder.device, dtype=torch.float32), 
                    emb=None
                ) # 64x1x8x8
                print(condition.shape)
                condition = condition.unsqueeze(0)
                print(condition.shape)

                sample_img = pl_module.sample(
                    num_samples=1, 
                    img_size=self.noise_shape, 
                    condition=condition
                ).detach()
                # => 1, 4, 128, 128

                # sample_img = sample_img.squeeze(0).reshape(8, 64, 16, 16)
                # # => 64, 2, 32, 32
                # sample_img = sample_img.permute(1, 0, 2, 3)

                sample_img = reverse_spatial_stack(sample_img, (16, 16), index_channel=False).squeeze(0)

                sample_img = pl_module.latent_embedder.decode(sample_img, emb=None)
                # => 64x2x128x128

                # selecting subset of the volume to display
                sample_img = sample_img[::4, ...] # 64 // 4 = 16

                if self.save:
                    save_image(sample_img[:, 0, None], '{}/sample_images_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)
                    save_image(sample_img[:, 1, None], '{}/sample_masks_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)

                sample_img = torch.cat([
                    torch.hstack([img for img in sample_img[:, idx, ...]]) for idx in range(sample_img.shape[1])
                ], dim=0)
                sample_img = sample_img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        sample_img.cpu().numpy(), 
                        caption='Epoch : {}'.format(trainer.current_epoch)
                    )
                })
