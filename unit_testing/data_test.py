import torch
from modules.data import BRATSDataModule
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # --------------- Settings --------------------
    # current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # path_run_dir = Path.cwd() / 'runs' / str(current_time)
    # path_run_dir.mkdir(parents=True, exist_ok=True)
    # gpus = [0] if torch.cuda.is_available() else None
    
    # logger = wandb_logger.WandbLogger(
    #     project='slice-baed-latent-diffusion', 
    #     name='first-stage',
    #     save_dir=path_run_dir
    #     # id='24hyhi7b',
    #     # resume="must"
    # )

    # ------------ Load Data ----------------
    print('Data prepare ...')
    datamodule = BRATSDataModule(
        data_dir='./data/brats_preprocessed.npy',
        train_ratio=0.8,
        batch_size=32,
        num_workers=32,
        shuffle=True,
        horizontal_flip=0.5,
        vertical_flip=0.5,
        rotation=[0, 90],
        random_crop_size=(96, 96),
        dtype=torch.float32,
        slice_wise=True
    )

    datamodule.setup()
    train_laoder = datamodule.train_dataloader()

    batch = next(iter(train_laoder))

    print(batch[0].shape, batch[1].shape)
    print(batch[0].max(), batch[0].min())
    print(batch[1])

    for i in range(16):
        plt.subplot(2, 8, i+1)
        plt.imshow(batch[0][i, 0, :, :], cmap='gray')
        plt.axis('off')
    plt.show()