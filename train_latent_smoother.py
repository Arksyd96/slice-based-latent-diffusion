import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime
import torch 
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

from modules.models.embedders.latent_smoother import LatentSmoother
from modules.models.embedders.latent_embedders import VAE
import wandb
from torchvision.utils import save_image

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from modules.data import BRATSDataModule

import os
os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'


class ReconstructionCallback(pl.Callback):
    def __init__(self, log_every_n_epochs = 1) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if trainer.global_rank == 0 and (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            with torch.no_grad():
                batch = trainer.train_dataloader.dataset.sample(1)
                x = batch[0].to(pl_module.device, torch.float32)
                z = pl_module.encode_as_2d_latent(x)
                z_hat, _ = pl_module.forward(z)
                x_hat = pl_module.decode_as_volume(z_hat) # []

                # selecting subset of the volume to display
                x_hat = x_hat.permute(0, 4, 1, 2, 3).squeeze(0)
                x_hat = x_hat[::4, ...] # 64 // 4 = 16

                x_hat = torch.cat([
                    torch.hstack([img for img in x_hat[:, idx, ...]]) for idx in range(x_hat.shape[1])
                ], dim=0)

                x_hat = x_hat.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        x_hat.cpu().numpy(), 
                        caption='{}'.format(trainer.current_epoch)
                    )
                })


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # --------------- Data --------------------
    datamodule = BRATSDataModule(
        train_dir       = './data/second_stage_dataset_128x128_100.npy',
        train_ratio     = 0.95,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 6,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = False
    )

    # ------------ Initialize Latent Space  ------------
    latent_embedder_checkpoint = './runs/first_stage-2023_08_11_230709 (best AE so far + mask)/last.ckpt'
    latent_embedder = VAE.load_from_checkpoint(latent_embedder_checkpoint)

    # ------------ Initialize Pipeline ------------
    model = LatentSmoother(
        channels = 8,
        latent_embedder = latent_embedder
    )

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/LatentSmoother-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    # --------------- Logger --------------------
    logger = wandb_logger.WandbLogger(
        project = 'slice-based-latent-diffusion', 
        name    = 'LatentSmoother [Bx2x128x128]',
        save_dir = save_dir
    )

    # -------------- Training Initialization ---------------
    recon_callback = ReconstructionCallback()

    checkpointing = ModelCheckpoint(
        dirpath=str(save_dir), # dirpath
        monitor=None,
        every_n_epochs=50,
        save_last=True,
        save_top_k=1
    )

    trainer = Trainer(
        logger      = logger,
        # strategy    = 'ddp_find_unused_parameters_true',
        # devices     = 4,
        #num_nodes   = 1,  
        precision   = 32,
        accelerator = 'gpu',
        # default_root_dir = save_dir,
        enable_checkpointing = True,
        log_every_n_steps = 1, 
        min_epochs = 100,
        max_epochs = 1000,
        num_sanity_val_steps = 0,
        # fast_dev_run = 10,
        callbacks=[recon_callback, checkpointing]
    )
    
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=datamodule)


