import torch
import argparse
import sys

from modules.models.embedders.latent_embedders import VAEGAN, VAE
from modules.models.pipelines.diffusion_pipeline import DiffusionPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True, help='Path to the diffusion model')
    parser.add_argument('-ac', '--ae-class', type=str, required=True, choices=['VAEGAN', 'VAE'], help='Class of the associated latent autoencoder')
    parser.add_argument('-ap', '--ae-path', type=str, required=True, help='Path to the associated latent autoencoder')
    parser.add_argument('-v', '--std-norm', type=float, default=1.0, help=
                        'Standard deviation of the encoded latents in order to denormalize sampled latents')
    parser.add_argument('-n', '--num-samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('-s', '--noise-shape', type=int, nargs='+', default=[1, 16, 16], help='Shape of the noise to sample')
    parser.add_argument('--ddim', action='store_true', help='Whether to use DDIM or not', default=False)
    parser.add_argument('--ddim-steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--condition', type=float, nargs='+', default=None, help='Condition to use for sampling [v_0, v_1, ..., v_n]')
    parser.add_argument('--save-volume', action='store_true', help='Save the generated volume')
    parser.add_argument('--save-png', action='store_true', help='Save the generated volume as sequence of PNGs')
    parser.add_argument('--save-path', type=str, default='./', help='Path to save the generated volume & PNG')
    parser.add_argument('--save-nifti', action='store_true', help='Save the generated volume as Nifti')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    args = parser.parse_args()

    # get a class from a string
    autoencoder = getattr(sys.modules[__name__], args.ae_class)
    autoencoder = autoencoder.load_from_checkpoint(args.ae_path, time_embedder=None)

    diffuser = DiffusionPipeline.load_from_checkpoint(args.model_path, latent_embedder=autoencoder, std_norm=args.std_norm)

    device = torch.device(args.device)
    autoencoder = autoencoder.to(device)
    diffuser = diffuser.to(device)
    diffuser.eval()

    if args.condition is not None:
        condition = torch.tensor(args.condition, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        condition = None

    with torch.no_grad():
        sample_volume = diffuser.sample(
            num_samples=args.num_samples, 
            img_size=args.noise_shape,
            condition=condition,
            use_ddim=args.ddim,
            steps=args.ddim_steps if args.ddim else 1000
        ).detach()
        if args.noise_shape.__len__() == 3:
            # slice-based latent diffusion; permute depth to batch for decoding
            sample_volume = sample_volume.permute(0, 2, 1, 3, 4) # [N, C, D, H, W]

        sample_volume = sample_volume.mul(torch.tensor(args.std_norm, dtype=torch.float32, device=device))
        
        samples = []
        for idx in range(args.num_samples):
            sample = sample_volume[idx, ...].permute(1, 0, 2, 3) # [D, C, H, W]
            sample = diffuser.latent_embedder.decode(sample, emb=None).unsqueeze(0) # [1, D, C, H, W]
            samples.append(sample)

        samples = torch.cat(samples, dim=0) # [N, D, C, H, W]

    if args.save_png:
        from torchvision.utils import save_image
        for idx in range(args.num_samples):
            save_image(samples[idx, ...], '{}/sample_{}.png'.format(args.save_path, idx), nrow=samples.shape[1], normalize=True)

    samples = samples.permute(0, 2, 1, 3, 4) # [N, C, D, H, W]
    if args.save_volume:
        import numpy as np
        np.save('{}/samples.npy'.format(args.save_path), samples.cpu().numpy())
    
    if args.save_nifti:
        import nibabel as nib
        for idx in range(args.num_samples):
            volume = samples[idx, ...].squeeze().cpu().numpy()
            volume = volume.transpose(1, 2, 0)
            nib.save(nib.Nifti1Image(volume, np.eye(4)), '{}/sample_{}.nii.gz'.format(args.save_path, idx))

    print('Done!')
