import torch
import numpy as np
from tqdm import tqdm

from modules.models.embedders.latent_embedders import VAE
from modules.data import BRATSDataModule

from datetime import datetime
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

from modules.models.pipelines import DiffusionPipeline
from modules.models.estimators import UNet
from modules.models.noise_schedulers import GaussianNoiseScheduler
from modules.models.embedders import TimeEmbbeding
from modules.models.embedders.latent_embedders import VAEGAN, VAE
from modules.models.embedders.cond_embedders import ConditionMLP
from modules.loggers import ImageGenerationLogger

from wgan import WGAN

if __name__ == '__main__':

    wgan = WGAN(
        in_channels=2,
        out_channels=2,
        spatial_dims=3,
        emb_channels=4,
        hid_chs=[32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        lr=1e-4,
        lr_scheduler=None,
        dropout=0.0,
        use_res_block=False,
        learnable_interpolation=True,
        use_attention='none'
    ).to('cuda')

    x = torch.randn(1, 2, 192, 192, 96).to('cuda')
    o = wgan.discriminator(x)
    print(o.shape)

    x = wgan.generator(o)
    print(x.shape)

    # for idx, block in enumerate(wgan.decoders):
    #     x = block(x)
    #     print(idx, x.shape) 

    # print(x.shape)