import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime
import numpy as np
import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

# from modules.data.datamodules import SimpleDataModule
from modules.models.pipelines import DiffusionPipeline
from modules.models.estimators import UNet
from modules.models.noise_schedulers import GaussianNoiseScheduler
from modules.models.embedders import TimeEmbbeding
from modules.models.embedders.latent_embedders import VAEGAN, VAE
from modules.models.embedders.cond_embedders import ConditionMLP
from modules.loggers import ImageGenerationLogger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from modules.data import BRATSDataModule

import os
os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/diffusion-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project='slice-based-latent-diffusion', 
        name='diffusion-training (3D + test maison)',
        save_dir=save_dir,
        # id='24hyhi7b',
        # resume="must"
    )

    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_240x240.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 4,
        num_workers     = 4,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32
    )


    # ------------ Initialize Model ------------
    cond_embedder = None 
    # cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'in_features': 12, 
        'out_features': 512, 
        'hidden_dim': 256
    }
 

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': 512 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 4, # takes also the index channel
        'out_ch': 4,  
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


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
    # latent_embedder_checkpoint = './runs/first_stage-2023_08_11_230709 (best AE so far + mask)/epoch=489-step=807030.ckpt'
    
    latent_embedder_checkpoint = './runs/first_stage-2023_09_27_115636/last.ckpt'
    latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint, time_embedder=None)

    mask_embedder_checkpoint = './runs/mask-embedder-2023_09_30_144008/last.ckpt'
    mask_embedder = latent_embedder.load_from_checkpoint(mask_embedder_checkpoint, **cond_embedder_kwargs)
   
    # ------------ Initialize Pipeline ------------
    # pipeline = DiffusionPipeline.load_from_checkpoint(
    #     './runs/diffusion-2023_09_14_165250/last.ckpt',
    #     latent_embedder=latent_embedder,
    #     std_norm = np.mean([1.0800604820251465, 0.6785210371017456]) # std of images and std of masks
    # )

    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        mask_embedder=mask_embedder,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.0, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        # std_norm = 0.5601330399513245
        std_norm = np.mean([1.0800604820251465, 0.6785210371017456])
    )
    

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath=str(save_dir), # dirpath
        monitor=None,
        every_n_epochs=50,
        save_last=True,
        save_top_k=1
    )

    image_logger = ImageGenerationLogger(
        noise_shape=(4, 128, 30, 30),
        save_dir=str(save_dir),
        save_every_n_epochs=10,
        save=True
    )

    trainer = Trainer(
        logger      = logger,
        # strategy    = 'ddp_find_unused_parameters_true',
        # devices     = 4,
        # num_nodes   = 2,  
        precision   = 32,
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 1500,
        num_sanity_val_steps = 0,
        # fast_dev_run = 10,
        callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=datamodule)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)


