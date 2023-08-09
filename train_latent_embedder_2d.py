import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path
from datetime import datetime

import torch 
import numpy as np
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None
    
    logger = wandb_logger.WandbLogger(
        project='slice-baed-latent-diffusion', 
        name='first-stage',
        save_dir=path_run_dir
        # id='24hyhi7b',
        # resume="must"
    )

    # ------------ Load Data ----------------
    print('preparing data...')
    data = np.load('./brats_preprocessed.npy', allow_pickle=True)
    data = data[:, 0, None]
    
    norm = lambda data: (2 * data - data.min() - data.max()) / (data.max() - data.min())
    for idx in range(data.shape[0]):
        data[idx] = norm(data[idx]).astype(np.float32)

    # keeping track on slice positions for positional embedding
    N, C, W, H, D = data.shape
    slice_positions = torch.arange(D)[None, :].repeat(N, 1)
    slice_positions = slice_positions.flatten()

    # merging depth and batch dimension
    data = data.transpose(0, 4, 1, 2, 3)
    data = data.reshape(-1, C, W, H)

    print(data.shape, slice_positions.shape)
    print(data.max(), data.min())
        
    # train test split
    tr_ds = IdentityDataset(data[:700], slice_positions[:700])
    val_ds = IdentityDataset(data[700:800], slice_positions[700:800])
    test_ds = IdentityDataset(data[800:], slice_positions[800:])
    
    dm = SimpleDataModule(
        ds_train = tr_ds,
        ds_test=test_ds,
        ds_val=val_ds,
        batch_size=8,
        num_workers=6,
        pin_memory=True
    )

    # ------------ Initialize Model ------------
    model = VAE(
        in_channels=1, 
        out_channels=1, 
        emb_channels=2,
        spatial_dims=2,
        hid_chs =    [ 64, 128, 256,  512], 
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,    2],
        deep_supervision=1,
        use_attention= 'none',
        loss = torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6
    )

    # model.load_pretrained(Path.cwd()/'runs/2022_12_01_183752_patho_vae/last.ckpt', strict=True)

    # model = VAEGAN(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=8,
    #     spatial_dims=2,
    #     hid_chs =    [ 64, 128, 256,  512],
    #     deep_supervision=1,
    #     use_attention= 'none',
    #     start_gan_train_step=-1,
    #     embedding_loss_weight=1e-6
    # )

    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_11_25_082209_chest_vae/last.ckpt')
    # model.load_pretrained(Path.cwd()/'runs/2022_11_25_232957_patho_vaegan/last.ckpt')


    # model = VQVAE(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention = 'none',
    # )


    # model = VQGAN(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     start_gan_train_step=-1,
    #     pixel_loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention='none',
    # )
    
    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_12_13_093727_patho_vqvae/last.ckpt')
    

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    
    trainer = Trainer(
        accelerator='gpu',
        logger=logger,
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1, 
        # auto_lr_find=False,
        # limit_train_batches=1000,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=0
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.save_dir, checkpointing.best_model_path)
