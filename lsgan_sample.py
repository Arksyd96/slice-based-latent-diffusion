import torch
from train_lsgan import LSGAN
import matplotlib.pyplot as plt


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

    sample = model.generator(torch.randn(8, 2048).to('cuda'))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(sample[i, 0, :, :, 48].detach().cpu().numpy(), cmap='gray')
    plt.show()