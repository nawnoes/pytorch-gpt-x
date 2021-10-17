import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from arg import ModelConfig
from dataset import GPTXDataset
from model.transformer import GPTX
from transformers import BertTokenizer
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.plugins import DeepSpeedPlugin


def build_dataloader(dataset, batch_size, train_rate=0.8,shuffle=True):
  train_data_len = int(len(dataset) * train_rate)
  valid_data_len = len(dataset) - train_data_len

  train_data, valid_data = random_split(dataset, (train_data_len, valid_data_len))

  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
  valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)

  return train_dataloader, valid_dataloader


if __name__=='__main__':
  torch.manual_seed(9)
  base_path = '..'

  # Config
  config_path = f'{base_path}/config.json'
  config = ModelConfig(config_path=config_path).get_config()

  # Tokenizer
  tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

  # Dataset
  dataset = GPTXDataset(tokenizer, config.max_seq_len, config.data_path)
  train_dataloader, valid_dataloader = build_dataloader(dataset, config.batch_size,0.9)

  model = GPTX(
        num_tokens=tokenizer.vocab_size,
        dim=config.dim,
        depth=config.depth,
        heads=config.n_head,
        max_seq_len=config.max_seq_len, # AxialPositionalEmbedding을 위한 (79,64) 값 and max_len/(bucket_size*2) == 0 이어야한다. 현재 bucket_size = 64
  )

  checkpoint_callback = ModelCheckpoint(
    dirpath=config.checkpoint_path,
    filename=f"{config.model_name}"+"{epoch}-{step}",
    every_n_train_steps=config.ckpt_step,
    save_top_k = 2
  )

  # logger
  logger = TensorBoardLogger('tb_logs', name=config.model_name)

  # Trainer
  trainer = pl.Trainer(gpus=config.gpu,
                       plugins=config.deepspeed_plugin,
                       precision=config.precision,
                       logger=logger,
                       accumulate_grad_batches=config.gradient_accumulation_steps,
                       max_epochs=config.epochs
                       )

  trainer.fit(model,train_dataloader=train_dataloader,val_dataloaders=valid_dataloader)