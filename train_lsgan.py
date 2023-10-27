import numpy as np
import torch
import torch.nn.functional as F
import os
from datetime import datetime
from modules.data import BRATSDataModule
import wandb
from pytorch_lightning.loggers import wandb as wandb_logger
from torchvision.utils import save_image

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


########### Architecture ##############

# class Discriminator(nn.Module):
#     def __init__(self, input_shape, in_channels, num_channels = 512):
#         super(Discriminator, self).__init__()

#         self.num_channels = num_channels
#         self.backbone = nn.Sequential(
#             *self.conv_block(in_channels, num_channels // 16, kernel_size=4, stride=2, padding=1, norm=False),
#             *self.conv_block(num_channels // 16, num_channels // 8, kernel_size=4, stride=2, padding=1),
#             *self.conv_block(num_channels // 8, num_channels // 4, kernel_size=4, stride=2, padding=1),
#             *self.conv_block(num_channels // 4, num_channels // 2, kernel_size=4, stride=2, padding=1),
#             *self.conv_block(num_channels // 2, num_channels, kernel_size=4, stride=2, padding=1)
#         )

#         self.dense_shape = np.array(input_shape) // (2 ** 5)
#         self.fc = nn.Sequential(
#             nn.Linear(num_channels * np.prod(self.dense_shape), 1)
#         )

#     def conv_block(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, norm=True):
#         return nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm),
#             nn.BatchNorm3d(out_channels) if norm else nn.Identity(),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.view(-1, np.prod(self.dense_shape) * self.num_channels)
#         x = self.fc(x)
#         return x
    
# class Generator(nn.Module):
#     def __init__(self, input_shape, latent_dim, out_channels, num_channels = 512):
#         super(Generator, self).__init__()
#         self.num_channels = num_channels
#         self.latent_dim = latent_dim
#         self.latent_shape = np.array(input_shape) // (2 ** 5)

#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, self.num_channels * np.prod(self.latent_shape)),
#             nn.BatchNorm1d(self.num_channels * np.prod(self.latent_shape)),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.backbone = nn.Sequential(
#             *self.transposed_conv_block(num_channels, num_channels // 2, kernel_size=4, stride=2, padding=1),
#             *self.transposed_conv_block(num_channels // 2, num_channels // 4, kernel_size=4, stride=2, padding=1),
#             *self.transposed_conv_block(num_channels // 4, num_channels // 8, kernel_size=4, stride=2, padding=1),
#             *self.transposed_conv_block(num_channels // 8, num_channels // 16, kernel_size=4, stride=2, padding=1),
#             *self.transposed_conv_block(num_channels // 16, num_channels // 16, kernel_size=4, stride=2, padding=1)
#         )

#         self.output_conv = nn.Conv3d(num_channels // 16, out_channels, kernel_size=3, stride=1, padding=1)

#     def transposed_conv_block(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, norm=True, act = True):
#         return nn.Sequential(
#             nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm),
#             nn.BatchNorm3d(out_channels) if norm else nn.Identity(),
#             nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()
#         )
        
#     def forward(self, noise):
#         x = self.fc(noise)
#         x = x.view(-1, self.num_channels, *self.latent_shape)
#         x = self.backbone(x)
#         x = self.output_conv(x)
#         return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_channels = 512):
        super(Discriminator, self).__init__()        
        
        self.conv1 = nn.Conv3d(in_channels, num_channels // 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(num_channels // 16, num_channels//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(num_channels//8)
        self.conv3 = nn.Conv3d(num_channels//8, num_channels//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(num_channels//4)
        self.conv4 = nn.Conv3d(num_channels//4, num_channels//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(num_channels//2)
        self.conv5 = nn.Conv3d(num_channels//2, num_channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(num_channels)
        
        self.conv6 = nn.Conv3d(num_channels, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6

        return output.mean(dim=(2, 3, 4))
    

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_channels = 512):
        super(Discriminator, self).__init__()        
        
        self.conv1 = nn.Conv3d(in_channels, num_channels // 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(num_channels // 16, num_channels//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(num_channels//8)
        self.conv3 = nn.Conv3d(num_channels//8, num_channels//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(num_channels//4)
        self.conv4 = nn.Conv3d(num_channels//4, num_channels//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(num_channels//2)
        self.conv5 = nn.Conv3d(num_channels//2, num_channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(num_channels)
        
        self.conv6 = nn.Conv3d(num_channels, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6

        return output
    

class Generator(nn.Module):
    def __init__(self, latent_dim, out_channels, num_channels = 512):
        super(Generator, self).__init__()
        _c = num_channels
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(self.latent_dim, 512 * 6 * 6 * 3)
        self.bn1 = nn.BatchNorm3d(_c)
        
        self.tp_conv2 = nn.Conv3d(_c, _c // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c // 2)
        
        self.tp_conv3 = nn.Conv3d(_c // 2, _c // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(_c // 4)
        
        self.tp_conv4 = nn.Conv3d(_c // 4, _c // 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(_c // 8)

        self.tp_conv5 = nn.Conv3d(_c // 8, _c // 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(_c // 16)
        
        self.tp_conv6 = nn.Conv3d(_c // 16, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, noise):
        h = self.fc(noise)
        h = h.view(-1, 512, 6, 6, 3)
        h = F.relu(self.bn1(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)
        h = F.relu(self.bn5(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv6(h)
        output = F.tanh(h)

        return output

class LSGAN(pl.LightningModule):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        num_channels    = 512,
        latent_dim      = 2048, 
        lr              = 0.0002,
        clip_value      = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.lr = lr
        self.clip_value = clip_value

        self.discriminator = Discriminator(in_channels=in_channels, num_channels=1024)
        self.generator = Generator(latent_dim=latent_dim, out_channels=out_channels, num_channels=num_channels)

        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

        self.adversarial_loss = torch.nn.MSELoss().to(self.device)

        self.save_hyperparameters()
        self.automatic_optimization = False

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def training_step(self, batch, batch_idx):
        real_x, = batch
        g_optimizer, d_optimizer = self.optimizers()
            
        real_y = torch.ones(real_x.size(0), 1) - torch.randn(real_x.size(0), 1) * 0.05
        fake_y = torch.randn(real_x.size(0), 1) * 0.05
        
        real_y = real_y.to(real_x.device, dtype=real_x.dtype)
        fake_y = fake_y.to(real_x.device, dtype=real_x.dtype)

        # --------------------- Train Generator ---------------------
        g_optimizer.zero_grad(set_to_none=True)

        z = torch.randn(real_x.size(0), self.latent_dim).to(real_x.device, dtype=real_x.dtype)
        fake_x = self.generator(z)
        g_loss = self.adversarial_loss(self.discriminator(fake_x).mean(), real_y)

        self.manual_backward(g_loss)
        if self.clip_value is not None:            
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_value)
        g_optimizer.step()

        # --------------------- Train Discriminator ---------------------
        d_optimizer.zero_grad(set_to_none=True)

        d_real_loss = self.adversarial_loss(self.discriminator(real_x).mean(), real_y)
        d_fake_loss = self.adversarial_loss(self.discriminator(fake_x.detach()).mean(), fake_y)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        self.manual_backward(d_loss)
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_value)
        d_optimizer.step()

        #############################################

        # logging
        self.log('d_real_loss', d_real_loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('d_fake_loss', d_fake_loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('d_loss', d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('g_loss', g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_optimizer, d_optimizer], []
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(device=self.device)
        return self.generator(z)
    

class LSGANGenerator(pl.Callback):
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
                sample_img = sample_img[0].permute(3, 0, 1, 2)
            
                # selecting subset of the volume to display
                sample_img = sample_img[::4, ...] # 64 // 4 = 16

                if self.save:
                    save_image(sample_img[:, 0, None], '{}/sample_images_{}.png'.format(self.save_dir, trainer.current_epoch), normalize=True)

                sample_img = torch.cat([
                    torch.hstack([img for img in sample_img[:, idx, ...]]) for idx in range(sample_img.shape[1])
                ], dim=0)

                sample_img = sample_img.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        sample_img.cpu().numpy(), 
                        caption='[{}] min: {} - max: {} - mean: {}'.format(
                            trainer.current_epoch, sample_img.min(), sample_img.max(), sample_img.mean()
                        )#, format_condition(condition[0].cpu().numpy()))
                    )
                })

#############################################   
# Main script
#############################################

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/LSGAN-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project     = 'comparative-models', 
        name        = 'LSGAN (3D + mask)',
        save_dir    = save_dir
    )

    # ------------ Load Data ----------------
    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192_100.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 16,
        shuffle         = True,
        horizontal_flip = 0.2,
        vertical_flip   = 0.2,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = False
    )

    # ------------ Initialize Model ------------
    model = LSGAN(
        in_channels=2,
        out_channels=2,
        num_channels=512,
        latent_dim=2048,
        lr=0.0002,
        clip_value=5
    )

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath     = save_dir, # dirpath
        every_n_epochs = 1,
        save_last   = True,
        save_top_k  = 1,
        mode        = 'min',
    )
    
    image_logger = LSGANGenerator(
        num_samples = 1,
        log_every_n_epochs = 1, 
        save      = False,
        save_dir  = save_dir
    )

        
    trainer = Trainer(
        logger      = logger,
        strategy    = 'ddp_find_unused_parameters_true',
        devices     = 1,
        num_nodes   = 1,  
        precision   = 32,
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
