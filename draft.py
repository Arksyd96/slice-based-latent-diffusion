import torch
import numpy as np


if __name__ == '__main__':
    print('loading')
    data = np.load('./data/brats_preprocessed.npz')
    data = data['data']

    print('{}'.format(data.shape))
    print(np.unique(data[:, :, 1, ...]))