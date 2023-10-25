from datetime import datetime
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from torchvision.utils import save_image
from modules.data import BRATSDataModule

class ReshapeToMinus1x1(nn.Module):
    def forward(self, x):
        return x.view(-1, 1)

class WGAN(pl.LightningModule):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        latent_dim      = 1024, 
        lr              = 0.0002,
        n_critic        = 5,
        clip_value      = 0.01,
        lambda_gp       = 10,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.lambda_gp = lambda_gp

        self.generator = nn.Sequential(
            nn.ConvTranspose3d(self.latent_dim, 512, kernel_size=(6, 6, 3), stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256, affine=True),
            nn.ReLU(True),

            nn.Dropout(p=0.5),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64, affine=True),
            nn.ReLU(True),

            nn.Dropout(p=0.5),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32, affine=True),
            nn.ReLU(True),

            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=(12, 12, 6), stride=1, padding=0),
            ReshapeToMinus1x1()
        )

        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)

        self.save_hyperparameters()
        self.automatic_optimization = False

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=real_samples.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates) # (batch_size, 1)
        
        fake = torch.ones((real_samples.size(0), 1), device=real_samples.device)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        real_samples, = batch
        g_optimizer, d_optimizer = self.optimizers()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(self.n_critic):
            d_optimizer.zero_grad(set_to_none=True)

            # Sample fake volumes
            z = torch.randn(real_samples.size(0), self.latent_dim, 1, 1, 1).to(device=real_samples.device)
            fake_samples = self.generator(z)

            d_real_loss = self.discriminator(real_samples)
            d_fake_loss = self.discriminator(fake_samples)
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_samples, fake_samples)

            # Adversarial loss
            d_loss = (d_fake_loss.mean() - d_real_loss.mean()) + self.lambda_gp * gradient_penalty

            self.manual_backward(d_loss)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_value)
            d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        z = torch.randn(real_samples.size(0), self.latent_dim, 1, 1, 1).to(device=real_samples.device)
        fake_samples = self.generator(z)
        g_loss = -self.discriminator(fake_samples).mean()

        self.manual_backward(g_loss)
        nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_value)
        g_optimizer.step()

        # logging
        self.log('d_real_loss', d_real_loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('d_fake_loss', d_fake_loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('d_loss', d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('g_loss', g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_optimizer, d_optimizer], []
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim, 1, 1, 1).to(device=self.device)
        return self.generator(z)
    

class WGANLogger(pl.Callback):
    def __init__(
        self,
        num_samples,
        log_every_n_epochs = 5,
        save      = True,
        save_dir  = os.path.curdir
    ) -> None:
        super().__init__()
        self.save = save
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = '{}/images'.format(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            with torch.no_grad():
                sample_img = pl_module.sample(num_samples=self.num_samples).detach()
                sample_img = sample_img.permute(0, 4, 1, 2, 3).squeeze(0)
            
                # selecting subset of the volume to display
                sample_img = sample_img[::4, ...] # 64 // 4 = 16

                if self.save:
                    save_image(sample_img[:, 0, None], '{}/sample_images_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)

                sample_img = torch.cat([
                    torch.hstack([img for img in sample_img[:, idx, ...]]) for idx in range(sample_img.shape[1])
                ], dim=0)

                sample_img = sample_img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        sample_img.cpu().numpy(), 
                        caption='[{}]'.format(trainer.current_epoch)#, format_condition(condition[0].cpu().numpy()))
                    )
                })

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

if __name__ == "__main__":
    pl.seed_everything(42)

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/WGAN-GP-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project     = 'comparative-models', 
        name        = 'WGAN-GP (3D + mask)',
        save_dir    = save_dir
    )

    # ------------ Load Data ----------------
    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 32,
        shuffle         = True,
        horizontal_flip = 0.2,
        vertical_flip   = 0.2,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float16,
        include_radiomics = False
    )

    # ------------ Initialize Model ------------
    model = WGAN(
        in_channels     = 2,
        out_channels    = 2,
        latent_dim      = 2048,
        lr              = 0.0002,
        n_critic        = 4,
        clip_value      = 0.01
    )

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath     = save_dir, # dirpath
        monitor     = 'val/loss', # 'val/ae_loss_epoch',
        every_n_epochs = 1,
        save_last   = True,
        save_top_k  = 1,
        mode        = 'min',
    )
    
    image_logger = WGANLogger(
        num_samples = 1,
        log_every_n_epochs = 10, 
        save      = False,
        save_dir  = save_dir
    )

    ddp = DDPStrategy(process_group_backend='nccl')
        
    trainer = Trainer(
        logger      = logger,
        strategy    = ddp,
        devices     = 8,
        num_nodes   = 1,  
        precision   = 'bf16',
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        check_val_every_n_epoch = 1,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 4000,
        num_sanity_val_steps = 0,
        callbacks=[checkpointing, image_logger]
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=datamodule)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)

