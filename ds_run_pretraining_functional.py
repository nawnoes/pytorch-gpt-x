import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

import torch
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from transformers import BertTokenizer

import os
import json
import logging채ㅜ
from datetime import datetime
from dataset import GPT3Dataset
from arg import ModelConfig
from model.reformer_gpt_x import ReformerGPTX
from model.transformer_gpt_x import TransformerGPTX
from ds_util import get_argument_parser
import deepspeed

def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args

def build_dataloaders(config, dataset, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
    dataset_len = len(dataset)
    eval_len = int(dataset_len * train_test_split)
    train_len = dataset_len - eval_len
    train_dataset, eval_dataset = random_split(dataset, (train_len, eval_len))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=train_shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=eval_shuffle)
    logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                     eval_dataloader  size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

    return train_loader, eval_loader

def train_gpt3(config,
          model,
          train_dataloader,
          eval_dataloader,
          ):
    losses = {}
    global_steps = 0
    local_steps = 0
    step_loss = 0.0
    start_epoch = 0
    start_step = 0
    step_perplexity = 0.0

    # Load Checkpoint
    if os.path.isfile(f'{config.checkpoint_path}/{config.model_name}'):
        _, checkpoint_state_dict = model.load_checkpoint(config.checkpoint_path, config.model_name)
        start_epoch = checkpoint_state_dict['epoch']
        start_step = checkpoint_state_dict['step']

    model.train()
    # model.to(config.device)

    # Logging
    logging.info(f'{datetime.now()} | Moved model to: {config.device}')
    logging.info(f'{datetime.now()} | train_batch_size: {config.batch_size} | eval_batch_size: {config.batch_size}')
    logging.info(f'{datetime.now()} | Epochs: {config.epochs} | log_steps: {config.log_steps} | ckpt_steps: {config.ckpt_steps}')
    logging.info(f'{datetime.now()} | gradient_accumulation_steps: {config.gradient_accumulation_steps}')

    # Train
    # model.zero_grad()  # Reset gradients tensors
    for epoch in range(start_epoch, config.epochs): #tqdm(range(epochs), desc='Epochs', position=0):
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
            inputs, labels, inputs_mask = inputs.to(config.device), labels.to(config.device), inputs_mask.to(config.device)
            lm_logit, loss = model(inputs,labels,input_mask=inputs_mask)

            step_perplexity += torch.exp(loss)
            origin_loss = loss.item()

            loss = loss / config.gradient_accumulation_steps  # divide loss into gradient accumulation step
            model.backward(loss) # run backpropagation

            step_loss += origin_loss
            losses[global_steps] = origin_loss

            local_steps += 1
            global_steps += 1

            if model.is_gradient_accumulation_boundary():
                model.step()

            if global_steps % config.log_steps == 0:
                pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                step_loss = 0.0
                local_steps = 0
                step_perplexity = 0.0

            if global_steps % config.ckpt_steps == 0:
                save(config, model, epoch, losses, global_steps)
                logging.info(f'{datetime.now()} | Saved checkpoint to: {config.checkpoint_path}')
                with open(f'{config.log_dir}/{config.model_name}_train_results.json', 'w') as results_file:
                    json.dump(losses, results_file)
                    results_file.close()

        # Evaluate every epoch
        evaluate(eval_dataloader)
        model.train()
        start_step = 0

    save(config, model, epoch, losses, global_steps)

def evaluate(config, model, dataloader):
    model.eval()

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
        inputs, labels, inputs_mask = inputs.to(config.device), labels.to(config.device), inputs_mask.to(config.device)

        with torch.no_grad():
            lm_logit, loss = model(inputs, labels, input_mask=inputs_mask)

        tmp_eval_loss = loss
        tmp_perplexity = torch.exp(tmp_eval_loss)

        eval_loss += tmp_eval_loss.item()
        perplexity += tmp_perplexity.item()
        eval_steps += 1

        total_eval_loss = eval_loss/eval_steps
        total_perplexity= perplexity/eval_steps

        logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}')
        with open(f'{config.log_dir}/{config.model_name}_eval_results.txt', 'a+') as results_file:
            results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}\n')
            results_file.close()

def save(config, model, epoch, losses, train_step):
    checkpoint_state_dict = {
      'epoch': epoch,  # 현재 학습 epoch
      'losses': losses,  # Loss 저장
      'train_step': train_step,  # 현재 진행한 학습
    }
    model.save_checkpoint(config.checkpoint_path, config.model_name, checkpoint_state_dict)


def main():
    torch.manual_seed(9)
    args = get_arguments()

    print("DeepSpeed Args = {}".format(args))

    # Config
    config = ModelConfig(config_path=args.config).get_config()

    # Tokenizer
    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

    # Dataset
    dataset = GPT3Dataset(tokenizer, config.max_seq_len, config.data_path)

    # Logging
    logging.basicConfig(filename=f'{config.log_dir}/{config.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    # Model
    if 'reformer' in config.model_name:
        model = ReformerGPTX(
            num_tokens=tokenizer.vocab_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.n_head,
            max_seq_len=config.max_seq_len, # AxialPositionalEmbedding을 위한 (79,64) 값 and max_len/(bucket_size*2) == 0 이어야한다. 현재 bucket_size = 64
        )
    elif 'transformer' in config.model_name:
        model = TransformerGPTX(
            vocab_size= tokenizer.vocab_size,
            dim = config.dim,
            depth = config.depth,
            head_num= config.n_head,
            max_seq_len= config.max_seq_len,
        )
    model.cuda()

    # DeepSpeed initialize
    model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                  model=model,
                                                  model_parameters=model.parameters())
    # load data
    train_dataloader, eval_dataloader = build_dataloaders(config, dataset, train_test_split=0.1)

    # train model
    train_gpt3(config, model, train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()
