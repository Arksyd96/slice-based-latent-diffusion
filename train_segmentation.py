import os
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import wandb as wandb_logger
from modules.data import BRATSDataModule

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(seed)
    if debugging:
        torch.backends.cudnn.deterministic = True

################### UNET #####################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear

        self.inc = (DoubleConv(in_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if trilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, trilinear))
        self.up2 = (Up(256, 128 // factor, trilinear))
        self.up3 = (Up(128, 64 // factor, trilinear))
        self.up4 = (Up(64, 32, trilinear))
        self.outc = (OutConv(32, out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    # assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target) 


class LitUNet(pl.LightningModule):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        bilinear=True,
        lr=0.001
    ):
        super().__init__()
        self.lr = lr
        self.model = UNet(in_channels, out_channels, bilinear)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch.chunk(2, dim=1)
        x, y = x.type(torch.float32), y.type(torch.float32)
        y_hat = self.model(x)
        loss = dice_loss(torch.sigmoid(y_hat.squeeze(1)), y.squeeze(1).float())
        self.log('train_dice', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.chunk(2, dim=1)
        x, y = x.type(torch.float32), y.type(torch.float32)
        y_hat = self.model(x)
        loss = dice_loss(torch.sigmoid(y_hat.squeeze(1)), y.squeeze(1).float())
        self.log('val_dice', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch.chunk(2, dim=1)
        x, y = x.type(torch.float32), y.type(torch.float32)
        y_hat = self.model(x)
        loss = dice_loss(torch.sigmoid(y_hat.squeeze(1)), y.squeeze(1).float())
        self.log('test_dice', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def use_checkpointing(self):
        self.model.use_checkpointing()

##########################################################################################
os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

if __name__ == "__main__":
    global_seed(42, debugging=False)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')

    # --------------- Data --------------------
    train_dataset = np.load('./data/second_stage_dataset_192x192_100_train.npy')
    train_dataset[:, 1] = np.where(train_dataset[:, 1] == -1, 0, train_dataset[:, 1])
    val_dataset = np.load('./data/second_stage_dataset_192x192_100_eval.npy')
    val_dataset[:, 1] = np.where(val_dataset[:, 1] == -1, 0, val_dataset[:, 1])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    model = LitUNet(
        in_channels=1,
        out_channels=1,
        bilinear=True,
        lr=0.0001
    )

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/UNet-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project = 'comparative-models', 
        name    = 'UNet (Real data)',
        save_dir = save_dir
    )

    # --------------- Training --------------------

    trainer = Trainer(
        logger      = logger,
        strategy    = 'ddp',
        devices     = 1,
        num_nodes   = 1,  
        precision   = 32,
        accelerator = 'gpu',
        default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 10,
        max_epochs = 80,
        num_sanity_val_steps = 0,
        # fast_dev_run = 10,
        # callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, train_dataloader)

    # ---------------- Evaluate ----------------
    trainer.test(model, datamodule=val_dataloader)