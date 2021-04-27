import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from transformers import BertTokenizer
# from apex import amp

from fairseq.optim.adafactor import Adafactor
import os
import json
import logging
from datetime import datetime
from dataset import GPT3Dataset
from arg import ModelConfig
from model_gpt3 import ReformerGPT3

class ReformerGPT3Trainer(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 model_name,
                 checkpoint_path,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 log_dir='../logs',
                 fp16 = True):

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.log_dir = log_dir
        self.fp16 = fp16

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=eval_shuffle)
        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

        return train_loader, eval_loader

    def train(self,
              epochs,
              train_dataloader,
              eval_dataloader,
              optimizer,
              scheduler,
              log_steps,
              ckpt_steps,
              gradient_accumulation_steps=1):

        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0
        start_epoch = 0
        start_step = 0
        step_perplexity = 0.0

        # Load Checkpoint
        if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
            checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            global_steps = checkpoint['train_step']
            start_step = global_steps if start_epoch==0 else global_steps*self.train_batch_size % len(train_dataloader)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            amp.load_state_dict(checkpoint['amp'])

        self.model.train()
        self.model.to(self.device)

        # Logging
        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        # Train
        self.model.zero_grad()  # Reset gradients tensors
        for epoch in range(start_epoch, epochs): #tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            pb = tqdm(enumerate(train_dataloader),
                      desc=f'Epoch-{epoch} Iterator',
                      total=len(train_dataloader),
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            for step, batch in pb:
                if step < start_step:
                    continue
                inputs, labels, inputs_mask = batch
                inputs, labels, inputs_mask = inputs.to(self.device), labels.to(self.device), inputs_mask.to(self.device)
                lm_logit, loss = self.model(inputs,labels,input_mask=inputs_mask)

                step_perplexity += torch.exp(loss)
                origin_loss = loss.item()

                loss = loss / gradient_accumulation_steps  # divide loss into gradient accumulation step
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                step_loss += origin_loss
                losses[global_steps] = origin_loss

                local_steps += 1
                global_steps += 1

                if global_steps % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    self.model.zero_grad()

                if global_steps % log_steps == 0:
                    pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                    step_loss = 0.0
                    local_steps = 0
                    step_perplexity = 0.0

                if global_steps % ckpt_steps == 0:
                    self.save(epoch, self.model, optimizer, losses, global_steps)
                    logging.info(f'{datetime.now()} | Saved checkpoint to: {self.checkpoint_path}')
                    with open(f'{self.log_dir}/{self.model_name}_train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()

            # Evaluate every epoch
            self.evaluate(eval_dataloader)
            self.model.train()
            start_step = 0

        self.save(epoch, self.model, optimizer, losses, global_steps)

        return self.model

    def evaluate(self, dataloader):
        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader),
                                desc='Evaluating',
                                leave=True,
                                total=len(dataloader),
                                bar_format='{l_bar}{bar:10}{r_bar}'):

            inputs, labels, inputs_mask = batch
            inputs, labels, inputs_mask = inputs.to(self.device), labels.to(self.device), inputs_mask.to(self.device)

            with torch.no_grad():
                lm_logit, loss = self.model(inputs, labels, input_mask=inputs_mask)

            tmp_eval_loss = loss
            tmp_perplexity = torch.exp(tmp_eval_loss)

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            perplexity += tmp_perplexity.item()
            eval_steps += 1

            total_eval_loss = eval_loss/eval_steps
            total_perplexity= perplexity/eval_steps

            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}')
            with open(f'{self.log_dir}/{self.model_name}_eval_results.txt', 'a+') as results_file:
                results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}\n')
                results_file.close()

    def save(self, epoch, model, optimizer, losses, train_step):
        torch.save({
            'epoch': epoch,  # 현재 학습 epoch
            'model_state_dict': model.state_dict(),  # 모델 저장
            'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
            'losses': losses,  # Loss 저장
            'train_step': train_step,  # 현재 진행한 학습
            'amp': amp.state_dict()
        }, f'{self.checkpoint_path}/{self.model_name}.pth')


def main():
    torch.manual_seed(9)
    # base_path = '/content/drive/My Drive/Colab Notebooks/transformer-electra'
    base_path = '.'
    # base_path = '/Users/a60058238/Desktop/dev/workspace/transformer-electra'

    log_dir = f'{base_path}/logs'
    config_path = f'{base_path}/config.json'

    # Config
    config = ModelConfig(config_path=config_path).get_config()

    # Tokenizer
    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

    # Dataset
    dataset = GPT3Dataset(tokenizer, config.max_seq_len, config.data_path)

    model = ReformerGPT3(
        num_tokens=tokenizer.vocab_size,
        dim=config.dim,
        depth=config.depth,
        heads=config.n_head,
        max_seq_len=config.max_seq_len, # AxialPositionalEmbedding을 위한 (79,64) 값 and max_len/(bucket_size*2) == 0 이어야한다. 현재 bucket_size = 64
    )

    model.cuda()

    # optimizer = Adafactor(model.parameters())
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

    if config.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)


    # Pretraining Trainer
    trainer = ReformerGPT3Trainer(dataset, model, tokenizer,
                                  checkpoint_path=config.checkpoint_path,
                                  max_len=config.max_seq_len,
                                  model_name=config.model_name,
                                  train_batch_size=config.batch_size,
                                  eval_batch_size=config.batch_size,
                                  log_dir=log_dir,
                                  fp16=config.fp16
                                  )

    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

    trainer.train(epochs=config.epochs,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  optimizer=optimizer,
                  log_steps=config.log_steps,
                  ckpt_steps=config.ckpt_steps,
                  gradient_accumulation_steps=config.gradient_accumulation_steps)

if __name__ == '__main__':
    main()
