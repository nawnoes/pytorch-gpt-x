import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from arg import ModelConfig

if __name__=='__main__':
  torch.manual_seed(9)
  base_path = '..'

  config_path = f'{base_path}/config.j'
  config = ModelConfig(config_path=config_path).get_config()