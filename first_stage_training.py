import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from modules.data import BRATSDataModule
from modules.models.embedders.latent_embedders import VAE, VAEGAN
from modules.loggers import ImageReconstructionLogger

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/first_stage-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project = 'slice-based-latent-diffusion', 
        name    = 'first-stage VAE Mask (240x240)',
        save_dir = save_dir
    )

    # ------------ Load Data ----------------
    datamodule = BRATSDataModule(
        data_dir        = './data/brats_preprocessed.npy',
        train_ratio     = 0.95,
        batch_size      = 32,
        num_workers     = 32,
        shuffle         = True,
        horizontal_flip = 0.5,
        vertical_flip   = 0.5,
        rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        slice_wise      = True
    )


    # ------------ Initialize Model ------------
    # model = VAE(
    #     in_channels     = 1, 
    #     out_channels    = 1, 
    #     emb_channels    = 1,
    #     spatial_dims    = 3, # 2D or 3D
    #     hid_chs         = [32, 64, 128, 256], 
    #     kernel_sizes    = [3, 3, 3, 3],
    #     strides         = [1, 2, 2, 2],
    #     time_embedder   = None,
    #     deep_supervision = False,
    #     use_attention   = 'none', # ['none', 'none', 'none', 'spatial'],
    #     loss            = torch.nn.MSELoss,
    #     embedding_loss_weight = 1e-6,
    #     optimizer_kwargs = {'lr': 1e-5}
    # )

    model = VAEGAN(
        in_channels     = 2, 
        out_channels    = 2, 
        emb_channels    = 4,
        spatial_dims    = 2,
        hid_chs         = [128, 256, 512, 512],
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        time_embedder   = None,
        deep_supervision = False,
        use_attention   = ['none', 'none', 'none', 'spatial'],
        start_gan_train_step = 30001,
        embedding_loss_weight = 1e-6
    )
    

    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath     = save_dir, # dirpath
        monitor     = 'val/ae_loss_epoch',
        every_n_epochs = 10,
        save_last   = True,
        save_top_k  = 1,
        mode        = 'min',
    )
    
    image_logger = ImageReconstructionLogger(
        n_samples = 6,
        save      = True,
        save_dir  = save_dir,
        is_3d     = True
    )
        
    trainer = Trainer(
        logger      = logger,
        # strategy    = 'ddp',
        # devices     = 4,
        # num_nodes   = 2,  
        precision   = 32,
        accelerator = 'gpu',
        # gradient_clip_val=0.5,
        default_root_dir = save_dir,
        enable_checkpointing = True,
        check_val_every_n_epoch = 1,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 1000,
        num_sanity_val_steps = 0,
        callbacks=[checkpointing, image_logger]
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=datamodule)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)
