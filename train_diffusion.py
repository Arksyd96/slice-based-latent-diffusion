import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

from modules.models.pipelines import DiffusionPipeline
from modules.models.estimators import UNet
from modules.models.embedders import TimeEmbbeding
from modules.models.noise_schedulers import GaussianNoiseScheduler
from modules.loggers import ImageGenerationLogger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from modules.data import BRATSDataModule

import os
os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/DDPM-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        # project='slice-based-latent-diffusion',
        project='comparative-models', 
        name='DDPM (3D + mask)',
        save_dir=save_dir,
        # id='24hyhi7b',
        # resume="must"
    )

    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 6,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float16,
        include_radiomics = False
    )


    # ------------ Initialize Model ------------
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
        'in_ch': 2,
        'out_ch': 2,  
        'spatial_dims': 3,
        'hid_chs': [32, 64, 128, 256, 512],
        'kernel_sizes': [3, 3, 3, 3, 3],
        'strides': [1, 2, 2, 2, 2],
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
    # latent_embedder = VAE
    # latent_embedder_checkpoint = './runs/first_stage-2023_08_11_230709 (best AE so far + mask)/epoch=489-step=807030.ckpt'
    
    # latent_embedder_checkpoint = './runs/LDM-first-stage-2023_10_12_172125/last.ckpt'
    # latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint, time_embedder=None)

    # ------------ Initialize Pipeline ------------
    # pipeline = DiffusionPipeline.load_from_checkpoint(
    #     './runs/diffusion-2023_10_06_154034 (6 ch - 192x192x96 + mask + cond)/last.ckpt',
    #     latent_embedder=latent_embedder,
    #     std_norm = 0.8856033086776733
    # )

    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        # latent_embedder=latent_embedder,
        # mask_embedder=mask_embedder,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.0, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        # std_norm = 0.8856033086776733
        std_norm=None
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
        noise_shape=(2, 192, 192, 96),
        save_dir=str(save_dir),
        save_every_n_epochs=15,
        save=True
    )

    trainer = Trainer(
        logger      = logger,
        strategy    = 'ddp_find_unused_parameters_true',
        devices     = 4,
        num_nodes   = 2,  
        precision   = 16,
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 3000,
        num_sanity_val_steps = 0,
        # fast_dev_run = 10,
        callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=datamodule)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)


