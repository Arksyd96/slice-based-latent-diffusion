import numpy as np
import argparse
import torch
from tqdm import tqdm

from modules.models.embedders.latent_embedders import VAEGAN
from modules.data import BRATSDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True)
    parser.add_argument('-w', '--write', type=bool, default=False, required=False)
    parser.add_argument('-d', '--file-dir', type=str, default='.', required=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAEGAN.load_from_checkpoint(args.model_path).to(device)
    model.eval()

    datamodule = BRATSDataModule(
        data_dir        = './data/brats_preprocessed.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm',
        batch_size      = 16,
        num_workers     = 32,
        dtype           = torch.float32,
        slice_wise      = False,
        drop_channels   = [1] # image only
    )

    datamodule.setup()
    dataset = datamodule.train_dataset

    latents = []
    with torch.no_grad():
        for idx in tqdm(range(dataset.__len__()), position=0, leave=True, desc='Encoding ...'):
            x = dataset[idx][0].to(device)
            x_hat = model.encode(x, emb=None)
            latents.append(x_hat.detach().unsqueeze(0))
        
        latents = torch.cat(latents, dim=0)
        latents = latents.permute(0, 2, 1, 3, 4)    
        std = latents.std()

    print('std: {}'.format(std))

    if args.write:
        with open('{}/std.txt'.format(args.file_dir), 'w') as f:
            f.write('std: {}'.format(std.cpu().numpy()))
