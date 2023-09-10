import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

# from modules.data.datamodules import SimpleDataModule
from modules.models.pipelines import DiffusionPipeline
from modules.models.estimators import UNet
from modules.models.noise_schedulers import GaussianNoiseScheduler
from modules.models.embedders import TimeEmbbeding
from modules.models.embedders.cond_embedders import ConditionMLP
from modules.models.embedders.latent_embedders import VAE
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
        name='diffusion-training (3D avec mask embedder [Criann])',
        save_dir=save_dir
        # id='24hyhi7b',
        # resume="must"
    )

    datamodule = BRATSDataModule(
        data_dir        = './data/brats_preprocessed.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm',
        batch_size      = 8,
        num_workers     = 6,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        slice_wise      = False
    )


    # ------------ Initialize Model ------------
    cond_embedder = ConditionMLP 
    # cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'in_features': 4096,
        'out_features': 2048,
        'hidden_dim': 512
    }
 

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': 2048 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 4, 
        'out_ch': 4, 
        'spatial_dims': 2,
        'hid_chs': [256, 256, 512, 1024],
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
        'schedule_strategy': 'cosine'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
    # latent_embedder_checkpoint = './runs/first_stage-2023_08_11_230709 (best AE so far + mask)/epoch=489-step=807030.ckpt'
    latent_embedder_checkpoint = './runs/first_stage-2023_08_25_144308 (VAE 3 ch)/last.ckpt'
    mask_latent_embedder_checkpoint = './runs/mask-embedder-2023_09_09_192818-WLZR2X/last.ckpt'

    latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint, time_embedder=None)
    mask_latent_embedder = latent_embedder.load_from_checkpoint(mask_latent_embedder_checkpoint, time_embedder=None)
   
    # ------------ Initialize Pipeline ------------
    # pipeline = DiffusionPipeline.load_from_checkpoint(
    #     './runs/2023_08_16_113818 (continuity)/epoch=999-step=63000.ckpt', 
    #     latent_embedder=latent_embedder
    # )

    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        condition_latent_embedder=mask_latent_embedder,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.0, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False
    )
    
    # pipeline_old = pipeline.load_from_checkpoint('runs/2022_11_27_085654_chest_diffusion/last.ckpt')
    # pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath=str(save_dir), # dirpath
        monitor=None,
        every_n_epochs=50,
        save_last=True,
        save_top_k=1
    )

    image_logger = ImageGenerationLogger(
        noise_shape=(4, 128, 128),
        save_dir=str(save_dir),
        save_every_n_epochs=10,
        save=True
    )

    trainer = Trainer(
        logger      = logger,
        strategy    = 'ddp_find_unused_parameters_true',
        devices     = 4,
        num_nodes   = 2,  
        precision   = 32,
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 3000,
        num_sanity_val_steps = 0,
        callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=datamodule)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)


