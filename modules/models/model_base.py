
from pathlib import Path
import json

import torch 
import torch.nn.functional as F
import pytorch_lightning as pl

class VeryBasicModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._step_train = 0
        self._step_val = 0
        self._step_test = 0
        self.validation_step_outputs = []

    def forward(self, x_in):
        raise NotImplementedError

    def _step(self, batch, batch_idx, split, step):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch, batch_idx):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val)

    def test_step(self, batch, batch_idx):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def _epoch_end(self, outputs, split):
        return 
    
    def on_training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, version=0, **kwargs):
        path_version = 'lightning_logs/version_'+str(version)
        with open(Path(path_checkpoint_dir) / path_version/ 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir, version)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 

class BasicModel(VeryBasicModel):
    def __init__(self, 
                 optimizer=torch.optim.AdamW, 
                 optimizer_kwargs={'lr':1e-3, 'weight_decay':1e-2},
                 lr_scheduler= None, 
                 lr_scheduler_kwargs={},
                 ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]



    