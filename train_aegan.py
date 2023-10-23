import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger
from wgan import WGAN_GP, WGANImageGenerator

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from modules.data import BRATSDataModule

import os
os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/WGAN_GP-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        # project='slice-based-latent-diffusion',
        project='comparative-models', 
        name='WGAN-GP (3D + mask)',
        save_dir=save_dir,
        # id='24hyhi7b',
        # resume="must"
    )

    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 32,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = False
    )


    # ------------ Initialize Model ------------
    # aegan = WGAN_GP(
    #     input_shape=(192, 192, 96),
    #     in_channels=2,
    #     out_channels=2,
    #     spatial_dims=3,
    #     emb_channels=2,
    #     hid_chs=[32, 64, 128, 256, 512],
    #     kernel_sizes=[3, 3, 3, 3, 3],
    #     strides=[1, 2, 2, 2, 2],
    #     lr=0.002,
    #     lr_scheduler=None,
    #     dropout=0.0,
    #     use_res_block=False,
    #     learnable_interpolation=True,
    #     use_attention='none'
    # )

    aegan = WGAN_GP(
        latent_dim=1024,
        in_channels=2,
        out_channels=2,
        lr = 0.0002,
        g_iter=2,
        d_iter=1
    )
    
    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath=str(save_dir), # dirpath
        monitor=None,
        every_n_epochs=50,
        save_last=True,
        save_top_k=1
    )

    image_logger = WGANImageGenerator(sample_every_n_epochs=1, save=True, save_dir=save_dir)

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
        max_epochs = 3000,
        num_sanity_val_steps = 0,
        check_val_every_n_epoch=0,
        limit_val_batches=0,
        # fast_dev_run = 10,
        callbacks=[checkpointing, image_logger]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(aegan, datamodule=datamodule)

    # ------------- Save path to best model -------------
    aegan.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)


