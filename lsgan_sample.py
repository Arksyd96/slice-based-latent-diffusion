import torch
from train_lsgan import LSGAN
from tqdm import tqdm
import numpy as np


if "__main__" == __name__:
    model = LSGAN(
        in_channels=2,
        out_channels=2,
        num_channels=512,
        latent_dim=2048,
        lr=0.0002,
        clip_value=5
    )

    model.generator.load_state_dict(torch.load('./runs/LSGAN-2023_10_24_115407_last/G_W_iter1130.pth'))

    model.generator = model.generator.to('cuda')
    samples = []
    for idx in tqdm(range(100), position=0, leave=True): 
        sample = model.generator(torch.randn(1, 2048).to('cuda'))
        samples.append(sample.detach().cpu().numpy())

    samples = np.concatenate(samples, axis=0)
    np.save('samples/lsgan/samples.npy', samples)
