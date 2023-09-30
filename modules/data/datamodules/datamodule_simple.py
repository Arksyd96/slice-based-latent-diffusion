
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp 
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler

class SimpleDataModule(pl.LightningDataModule):

    def __init__(self,
                 ds_train: object,
                 ds_val:object =None,
                 ds_test:object =None,
                 batch_size: int = 1,
                 num_workers: int = mp.cpu_count(),
                 seed: int = 0, 
                 pin_memory: bool = False,
                 weights: list = None 
                ):
        super().__init__()

        self.ds_train = ds_train 
        self.ds_val = ds_val 
        self.ds_test = ds_test 

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed 
        self.pin_memory = pin_memory
        self.weights = weights

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.weights is not None:
            sampler = WeightedRandomSampler(self.weights, len(self.weights), generator=generator) 
        else:
            sampler = RandomSampler(self.ds_train, replacement=False, generator=generator)
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          sampler=sampler, generator=generator, drop_last=True)


    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, 
                                generator=generator, drop_last=False)
        else:
            raise AssertionError("A validation set was not initialized.")


    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, 
                            generator = generator, drop_last=False)
        else:
            raise AssertionError("A test test set was not initialized.")

   
   
    


        
      


    