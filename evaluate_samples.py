import argparse
import torch
import numpy as np
from skimage import metrics
import pytorch_ssim
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

def normalize(input_data, norm='centered-norm'):
    assert norm in ['centered-norm', 'z-score', 'min-max'], "Invalid normalization method"

    if norm == 'centered-norm':
        norm = lambda x: (2 * x - x.min() - x.max()) / (x.max() - x.min())
    elif norm == 'z-score':
        norm = lambda x: (x - x.mean()) / x.std()
    elif norm == 'min-max':
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return norm(input_data)

if __name__ == "__main__":
    target = torch.from_numpy(np.load('./data/second_stage_dataset_192x192_100.npy')).type(torch.float32)
    sbldm = torch.from_numpy(np.load('./samples/sbldm/SBLDM_samples_100.npy')).type(torch.float32)
    ldm = torch.from_numpy(np.load('./samples/ldm/samples.npy')).type(torch.float32)
    lsgan = torch.from_numpy(np.load('./samples/lsgan/samples.npy')).type(torch.float32)
    sbldm_1000 = torch.from_numpy(np.load('./samples/sbldm_1000/samples.npy')).type(torch.float32)

    # target = target.permute(0, 1, 4, 2, 3)
    rand_idx = np.random.permutation(target.shape[0])
    target = target[rand_idx]

    # sbldm = sbldm.permute(0, 1, 4, 2, 3)
    # ldm = ldm.permute(0, 1, 4, 2, 3)
    
    # sbldm = sbldm.clamp(-1, 1)
    # ldm = ldm.clamp(-1, 1)
    # lsgan = lsgan.clamp(-1, 1)

    # from -1, 1 to 0, 255
    target = target.add(1).div(2).clamp(0, 1)# .mul(255).clamp(0, 255).type(torch.uint8)
    sbldm = sbldm.add(1).div(2).clamp(0, 1)# .mul(255).clamp(0, 255).type(torch.uint8)
    ldm = ldm.add(1).div(2).clamp(0, 1)# .mul(255).clamp(0, 255).type(torch.uint8)
    lsgan = lsgan.add(1).div(2).clamp(0, 1)# .mul(255).clamp(0, 255).type(torch.uint8)
    sbldm_1000 = sbldm_1000.add(1).div(2).clamp(0, 1)# .mul(255).clamp(0, 255).type(torch.uint8)

    print('Target data shape: {}, Prediction data shape: {}'.format(target.shape, sbldm.shape))
    print('sbldm max: {}, min: {}'.format(sbldm.max(), sbldm.min()))
    print('ldm max: {}, min: {}'.format(ldm.max(), ldm.min()))
    print('lsgan max: {}, min: {}'.format(lsgan.max(), lsgan.min()))
    print('sbldm_1000 max: {}, min: {}'.format(sbldm_1000.max(), sbldm_1000.min()))
    print('target max: {}, min: {}'.format(target.max(), target.min()))
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, gaussian_kernel=False).to('cuda', dtype=torch.float32)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda', dtype=torch.float32)
    
    fid = FrechetInceptionDistance(feature=64, reset_real_features=False, normalize=True)
    fid.update(target[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=True)
    fid.update(sbldm[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=False)
    sbldm_fid_score = fid.compute().item()

    fid.update(target[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=True)
    fid.update(ldm[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=False)
    ldm_fid_score = fid.compute().item()

    fid.update(target[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=True)
    fid.update(lsgan[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=False)
    lsgan_fid_score = fid.compute().item()

    fid.update(target[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=True)
    fid.update(sbldm_1000[:, 0, None, :, :, 48].repeat(1, 3, 1, 1), real=False)
    sbldm_1000_fid_score = fid.compute().item()

    print('LDM FID: {}, SBLDM FID: {}, LSGAN: {}, SBLDM(1000) FID: {}'.format(ldm_fid_score, sbldm_fid_score, lsgan_fid_score, sbldm_1000_fid_score))

    ldm_ssim, sbldm_ssim, lsgan_ssim, sbldm_1000_ssim = [], [], [], []
    ldm_psnr, sbldm_psnr, lsgan_psnr, sbldm_1000_psrn = [], [], [], []
    for idx in tqdm(range(target.shape[0]), position=0, leave=True):
        # ldm_ssim.append(metrics.structural_similarity(ldm[idx].numpy(), target[idx].numpy(), data_range=255.0, channel_axis=0).item())
        # sbldm_ssim.append(metrics.structural_similarity(sbldm[idx].numpy(), target[idx].numpy(), data_range=255.0, channel_axis=0).item())
        # lsgan_ssim.append(metrics.structural_similarity(lsgan[idx].numpy(), target[idx].numpy(), data_range=255.0, channel_axis=0).item())

        # ldm_psnr.append(metrics.peak_signal_noise_ratio(ldm[idx].numpy(), target[idx].numpy(), data_range=255.0).item())
        # sbldm_psnr.append(metrics.peak_signal_noise_ratio(sbldm[idx].numpy(), target[idx].numpy(), data_range=255.0).item())
        # lsgan_psnr.append(metrics.peak_signal_noise_ratio(lsgan[idx].numpy(), target[idx].numpy(), data_range=255.0).item())

        ldm_ssim.append(ssim(ldm[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        sbldm_ssim.append(ssim(sbldm[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        lsgan_ssim.append(ssim(lsgan[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        sbldm_1000_ssim.append(ssim(sbldm_1000[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
    
        ldm_psnr.append(psnr(ldm[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        sbldm_psnr.append(psnr(sbldm[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        lsgan_psnr.append(psnr(lsgan[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())
        sbldm_1000_psrn.append(psnr(sbldm_1000[idx, 0, None, None].to('cuda'), target[idx, 0, None, None].to('cuda')).item())

    print('LDM SSIM: {}, SBLDM SSIM: {}, LSGAN SSIM: {}, SBLDM(1000) SSIM: {}'.format(np.mean(ldm_ssim), np.mean(sbldm_ssim), np.mean(lsgan_ssim), np.mean(sbldm_1000_ssim)))
    print('LDM PSNR: {}, SBLDM PSNR: {}, LSGAN PSNR: {}, SBLDM(1000) PSNR: {}'.format(np.mean(ldm_psnr), np.mean(sbldm_psnr), np.mean(lsgan_psnr), np.mean(sbldm_1000_psrn)))

    # psnr = PeakSignalNoiseRatio(data_range=2.0)
    # psnr_score = psnr(prediction, target)

    # ssim = metrics.structural_similarity(target, prediction, data_range=2.0, channel_axis=1)
    # psnr = metrics.peak_signal_noise_ratio(target, prediction, data_range=2.0)
    
    # ssim1 = pytorch_ssim.ssim3D(torch.from_numpy(target[:25]).to('cuda', dtype=torch.float32), torch.from_numpy(prediction[:25]).to('cuda', dtype=torch.float32))

    # m = []
    # for idx in range(25):
    #     m.append(
    #         pytorch_ssim.ssim3D(torch.from_numpy(target[idx, None]).to('cuda', dtype=torch.float32), torch.from_numpy(prediction[idx, None]).to('cuda', dtype=torch.float32)).item()
    #     )

    # m = np.array(m)

    # print(f"SSIM: {ssim1}")#, PSNR: {psnr}")
    # print(f"SSIM: {np.mean(m)}")#, PSNR: {psnr}"
    # print(f"SSIM: {ssim_score}, PSNR:.t {psnr_score}")


# SSIM: 0.4450096900679663, PSNR: 17.425916286698424
# SSIM: 0.5453276130324269, PSNR: 16.77954736138164