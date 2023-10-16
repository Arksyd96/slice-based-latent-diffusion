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

if __name__ == '__main__':

    cond_embedder = None
    # cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'in_features': 9, 
        'out_features': 512, 
        'hidden_dim': 256
    }
 

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': 512 # stable diffusion uses 4 * model_channels (model_channels is about 256)
    }
    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 6,
        'out_ch': 6,  
        'spatial_dims': 3,
        'hid_chs': [64, 128, 256, 512],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block': True,
        'use_attention': 'none',
    }

    model = UNet(**noise_estimator_kwargs).to('cuda')

    x = torch.randn(2, 6, 16, 16, 16).to('cuda')

    y = model(x)