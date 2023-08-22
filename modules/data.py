import numpy as np
import torch 
from torch import nn
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class IdentityDataset(torch.utils.data.Dataset):
    """
        Simple dataset that returns the same data (d0, d1, ..., dn)
    """
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]
    

class BRATSDataset(torch.utils.data.Dataset):
    """
        Images always as first argument, labels and other variables as last
        Transforms are applied on first argument only
    """
    def __init__(
        self,
        *data,
        transform       = None,
        resize          = None,
        horizontal_flip = None,
        vertical_flip   = None, 
        random_crop_size = None,
        rotation        = None,
        dtype           = torch.float32
    ):
        super().__init__()
        self.data = data

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(resize) if resize is not None else nn.Identity(),
                T.RandomHorizontalFlip(p=horizontal_flip) if horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip(p=vertical_flip) if vertical_flip else nn.Identity(),
                T.RandomCrop(random_crop_size) if random_crop_size is not None else nn.Identity(),
                T.RandomRotation(rotation) if rotation is not None else nn.Identity(),
                T.ConvertImageDtype(dtype),
                # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [self.transform(d[index]) if i == 0 else d[index] for i, d in enumerate(self.data)]
    
    def sample(self, n, transform=None):
        """ sampling randomly n samples from the dataset, apply or not the transform"""
        transform = self.transform if transform is not None else lambda x: x
        idx = np.random.choice(len(self), n)
        return [transform(d[idx]) if i == 0 else d[idx] for i, d in enumerate(self.data)]
    

class BRATSDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str, # should target an npy file, make sure to process your data first on an npy file
        train_ratio: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        resize = None,
        horizontal_flip = None,
        vertical_flip = None, 
        rotation = None,
        random_crop_size = None,
        dtype = torch.float32,
        slice_wise = False,
        include_seg = True,
        verbose = True
    ):
        super().__init__()
        self.dataset_kwargs = {
            "resize": resize,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip, 
            "random_crop_size": random_crop_size,
            "rotation": rotation,
            "dtype": dtype
        }
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.slice_wise = slice_wise
        self.train_ratio = train_ratio
        self.verbose = verbose
        self.include_seg = include_seg

    def setup(self, stage=None):
        data = torch.from_numpy(np.load(self.data_dir, allow_pickle=True))

        # data = data[:, 0, None, :, :, 32, None] ##
        if not self.include_seg:
            data = data[:, 0, None]
        
        # normalize to [0-1] (volume-wise normalization)
        norm = lambda data: (2 * data - data.min() - data.max()) / (data.max() - data.min())
        for idx in tqdm(range(data.shape[0]), desc="Normalizing data", position=0, leave=True):
            data[idx, 0] = norm(data[idx, 0])
            if self.include_seg:
                data[idx, 1] = torch.where(data[idx, 1] == 0, -1, data[idx, 1]) # labels [0, 1] => [-1, 1] for simplicity

        data = data.permute(0, 4, 1, 2, 3) # depth first

        if self.slice_wise:
            N, D, C, W, H = data.shape

            # merging depth and batch dimension
            data = data.reshape(N * D, C, W, H)

            # keeping track on slice positions for positional embedding
            slice_positions = torch.arange(D)[None, :].repeat(N, 1)
            slice_positions = slice_positions.flatten()

            # reducing number of empty slices only if included seg
            if self.include_seg:
                empty_slices_map = data[:, 0].mean(dim=(1, 2)) == -1 # only first channel (CARE!!)
                empty_slices_num = empty_slices_map.sum().item()
                num_to_set_false = int(empty_slices_num * 0.1)

                # randomly select 10% of the True values and set them to False (so we remove 90%)
                indices = torch.where(empty_slices_map == True)[0]
                indices_to_set_false = torch.randperm(empty_slices_num)[:num_to_set_false]
                empty_slices_map[indices[indices_to_set_false]] = False

                # removing selected empty slices
                data = data[~empty_slices_map]
                slice_positions = slice_positions[~empty_slices_map]

            # train val split
            train_images, val_images, train_positions, val_positions = train_test_split(
                data, slice_positions, train_size=self.train_ratio, random_state=42, stratify=slice_positions
            ) if self.train_ratio < 1 else (data, [], slice_positions, [])

            self.train_dataset = BRATSDataset(train_images, train_positions, **self.dataset_kwargs)
            self.val_dataset = BRATSDataset(val_images, val_positions)
        
        else:
            train_images, val_images = train_test_split(
                data, train_size=self.train_ratio, random_state=42
            ) if self.train_ratio < 1 else (data, [])

            self.train_dataset = BRATSDataset(train_images, **self.dataset_kwargs)
            self.val_dataset = BRATSDataset(val_images)

        log = """
        DataModule setup complete.
        Number of training samples: {}
        Number of validation samples: {}
        Data shape: {}
        Maximum: {}, Minimum: {}
        """.format(
            len(self.train_dataset),
            len(self.val_dataset),
            data.shape,
            data.max(),
            data.min()
        )

        if self.verbose: print(log)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = self.shuffle,
            pin_memory = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = True
        )
    
    


