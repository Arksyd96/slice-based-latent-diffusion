import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import os

from modules.models.embedders.latent_embedders import VAE
from modules.data import BRATSDataModule

from datetime import datetime
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import wandb as wandb_logger

from modules.models.pipelines import DiffusionPipeline
from modules.models.estimators import UNet
from modules.models.noise_schedulers import GaussianNoiseScheduler
from modules.models.embedders import TimeEmbbeding
from modules.models.embedders.latent_embedders import VAEGAN, VAE
from modules.models.embedders.cond_embedders import ConditionMLP
from modules.loggers import ImageGenerationLogger

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# 500 - version normale 100 reals
# 600 - version 50 reals
# 800 - 0 reals

# nnUNetv2_plan_and_preprocess -d 211 --verify_dataset_integrity
# nnUNetv2_preprocess -d 211
# nnUNetv2_train 211 3d_fullres all -tr nnUNetTrainerNoDA
# nnUNetv2_predict -i nnUNet/nnUNet_raw/Dataset211_MRI/imagesTs/ -o nnUNet/predicts/out_ldm_100 -d 211 -tr nnUNetTrainerNoDA -c 3d_fullres -f all 
# nnUNetv2_evaluate_folder.exe nnUNet/nnUNet_raw/Dataset211_MRI/labelsTs/ nnUNet/predicts/out_ldm_100/ -djfile nnUNet/predicts/out_ldm_100/dataset.json -pfile nnUNet/predicts/out_ldm_100/plans.json


if __name__ == '__main__':
    id = '211'
    name = 'MRI'

    if not os.path.exists('./nnUNet/nnUNet_raw/Dataset{}_{}'.format(id, name)):
        os.mkdir('./nnUNet/nnUNet_raw/Dataset{}_{}'.format(id, name))
        os.mkdir('./nnUNet/nnUNet_raw/Dataset{}_{}/imagesTr'.format(id, name))
        os.mkdir('./nnUNet/nnUNet_raw/Dataset{}_{}/labelsTr'.format(id, name))
        os.mkdir('./nnUNet/nnUNet_raw/Dataset{}_{}/imagesTs'.format(id, name))
        os.mkdir('./nnUNet/nnUNet_raw/Dataset{}_{}/labelsTs'.format(id, name))


    # loading data
    d = np.load('./data/second_stage_dataset_192x192_200.npy')

    baseline = d[:100]
    sbldm = np.load('./samples/ldm/samples.npy')[:100]

    baseline = np.concatenate([baseline, sbldm], axis=0)
    train_volumes, train_masks = baseline[:, 0, None], baseline[:, 1, None]
    train_masks = np.where(train_masks == -1, 0, train_masks)
    print('(Sanity check) Train uniques in mask: {}'.format(np.unique(train_masks)))

    test = d[100:]
    test_volumes, test_masks = test[:, 0, None], test[:, 1, None]
    test_masks = np.where(test_masks == -1, 0, test_masks)
    print('(Sanity check) Test uniques in mask: {}'.format(np.unique(test_masks)))

    for idx in tqdm(range(train_volumes.shape[0]), position=0, leave=True, desc='Creating training set'):
        volume, mask = train_volumes[idx, 0], train_masks[idx, 0] # 192, 192, 96
        nib.save(
            nib.Nifti1Image(volume, np.eye(4)), './nnUNet/nnUNet_raw/Dataset{}_{}/imagesTr/MRI_{}_0000.nii.gz'.format(
                id, name, str(idx).zfill(3)
            )
        )
        nib.save(
            nib.Nifti1Image(mask, np.eye(4)), './nnUNet/nnUNet_raw/Dataset{}_{}/labelsTr/MRI_{}.nii.gz'.format(
                id, name, str(idx).zfill(3)
            )
        )

    for idx in tqdm(range(test_volumes.shape[0]), position=0, leave=True, desc='Creating test set'):
        volume, mask = test_volumes[idx, 0], test_masks[idx, 0] # 192, 192, 96
        nib.save(
            nib.Nifti1Image(volume, np.eye(4)), './nnUNet/nnUNet_raw/Dataset{}_{}/imagesTs/MRI_{}_0000.nii.gz'.format(
                id, name, str(idx + train_volumes.shape[0]).zfill(3)
            )
        )
        nib.save(
            nib.Nifti1Image(mask, np.eye(4)), './nnUNet/nnUNet_raw/Dataset{}_{}/labelsTs/MRI_{}.nii.gz'.format(
                id, name, str(idx + train_volumes.shape[0]).zfill(3)
            )
        )

    print('Done!')


# usage: nnUNetv2_predict [-h] -i I -o O -d D [-p P] [-tr TR] -c C [-f F [F ...]] [-step_size STEP_SIZE] [--disable_tta] [--verbose] [--save_probabilities]
#                         [--continue_prediction] [-chk CHK] [-npp NPP] [-nps NPS] [-prev_stage_predictions PREV_STAGE_PREDICTIONS] [-num_parts NUM_PARTS]
#                         [-part_id PART_ID] [-device DEVICE]

# Use this to run inference with nnU-Net. This function is used when you want to manually specify a folder containing a trained nnU-Net model. This is useful when the
# nnunet environment variables (nnUNet_results) are not set.

# optional arguments:
#   -h, --help            show this help message and exit
#   -i I                  input folder. Remember to use the correct channel numberings for your files (_0000 etc). File endings must be the same as the training
#                         dataset!
#   -o O                  Output folder. If it does not exist it will be created. Predicted segmentations will have the same name as their source images.
#   -d D                  Dataset with which you would like to predict. You can specify either dataset name or id
#   -p P                  Plans identifier. Specify the plans in which the desired configuration is located. Default: nnUNetPlans
#   -tr TR                What nnU-Net trainer class was used for training? Default: nnUNetTrainer
#   -c C                  nnU-Net configuration that should be used for prediction. Config must be located in the plans specified with -p
#   -f F [F ...]          Specify the folds of the trained model that should be used for prediction. Default: (0, 1, 2, 3, 4)
#   -step_size STEP_SIZE  Step size for sliding window prediction. The larger it is the faster but less accurate the prediction. Default: 0.5. Cannot be larger than 1.
#                         We recommend the default.
#   --disable_tta         Set this flag to disable test time data augmentation in the form of mirroring. Faster, but less accurate inference. Not recommended.
#   --verbose             Set this if you like being talked to. You will have to be a good listener/reader.
#   --save_probabilities  Set this to export predicted class "probabilities". Required if you want to ensemble multiple configurations.
#   --continue_prediction
#                         Continue an aborted previous prediction (will not overwrite existing files)
#   -chk CHK              Name of the checkpoint you want to use. Default: checkpoint_final.pth
#   -npp NPP              Number of processes used for preprocessing. More is not always better. Beware of out-of-RAM issues. Default: 3
#   -nps NPS              Number of processes used for segmentation export. More is not always better. Beware of out-of-RAM issues. Default: 3
#   -prev_stage_predictions PREV_STAGE_PREDICTIONS
#                         Folder containing the predictions of the previous stage. Required for cascaded models.
#   -num_parts NUM_PARTS  Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one call predicts everything)
#   -part_id PART_ID      If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with num_parts - 1. So when you submit 5 nnUNetv2_predict
#                         calls you need to set -num_parts 5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible to make these run on
#                         separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)
#   -device DEVICE        Use this to set the device the inference should run with. Available options are 'cuda' (GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use
#                         this to set which GPU ID! Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!