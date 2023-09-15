import numpy as np
from nibabel import load
from nibabel.processing import resample_to_output
import os
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Path to the BRATS dataset folder')
    parser.add_argument('-n', '--n-samples', type=int, default=1000, help='Number of samples to load')
    parser.add_argument('-m', '--modalities', type=str, nargs='+', default=['t1', 't1ce', 't2', 'flair', 'seg'], help='Modalities to load')
    parser.add_argument('-t', '--target-shape', type=int, nargs='+', default=[128, 128, 64], help='Target shape to resize the volumes')
    parser.add_argument('-b', '--binarize', action='store_true', help='Binarize the segmentation mask')
    parser.add_argument('-s', '--save-path', type=str, default='./', help='Path to save the npy file')
    args = parser.parse_args()


    print('Loading dataset from NiFTI files...')
    placeholder = np.zeros(shape=(
        args.n_samples,
        args.modalities.__len__(), 
        args.target_shape[1], 
        args.target_shape[2], 
        args.target_shape[0]
    ))

    for idx, instance in enumerate(tqdm(os.listdir(args.data_path)[: args.n_samples], position=0, leave=True)):
        # loading models
        volumes = {}
        for _, m in enumerate(args.modalities):
            volumes[m] = load(os.path.join(args.data_path, instance, instance + f'_{m}.nii.gz'))

        # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
        orig_shape = volumes[args.modalities[0]].shape
        scale_factor = (orig_shape[0] / args.target_shape[1], # height
                        orig_shape[1] / args.target_shape[2], # width
                        orig_shape[2] / args.target_shape[0]) # depth

        # Resample the image using trilinear interpolation
        # Drop the last extra rows/columns/slices to get the exact desired output size
        for _, m in enumerate(args.modalities):
            volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
            volumes[m] = volumes[m][:args.target_shape[1], :args.target_shape[2], :args.target_shape[0]]

        # binarizing the mask (for simplicity), you can comment out this to keep all labels
        if args.binarize and 'seg' in args.modalities:
            volumes['seg'] = (volumes['seg'] > 0).astype(np.float32)

        # saving models
        for idx_m, m in enumerate(args.modalities):
            placeholder[idx, idx_m, :, :] = volumes[m]

    
    print('Saving dataset as npy file...')    
    # saving the dataset as a npy file
    np.save(args.save_path, placeholder)
    print('Done!')