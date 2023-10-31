import argparse
import numpy as np
from skimage import metrics

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True, help='Path to the target dataset')
    parser.add_argument('-p', '--prediction', type=str, required=True, help='Path to the prediction dataset')
    args = parser.parse_args()

    target = np.load(args.target)
    prediction = np.load(args.prediction)

    # rescaling predictions to -1, 1
    for idx in range(prediction.shape[0]):
        prediction[idx] = normalize(prediction[idx], norm='centered-norm')

    print('Target data shape: {}, Prediction data shape: {}'.format(target.shape, prediction.shape))
    print('max target: {}, max prediction: {}'.format(np.max(target), np.max(prediction)))
    print('min target: {}, min prediction: {}'.format(np.min(target), np.min(prediction)))

    ssim = metrics.structural_similarity(target, prediction, data_range=2.0, channel_axis=1)
    psnr = metrics.peak_signal_noise_ratio(target, prediction, data_range=2.0)

    print(f"SSIM: {ssim}, PSNR: {psnr}")


# SSIM: 0.4450096900679663, PSNR: 17.425916286698424



# SSIM: 0.5453276130324269, PSNR: 16.77954736138164