import sys
sys.path.append('../')

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import deepspeed
from common.dataset import GPTXDatasetV2
from common.arg import ModelConfig
from model.pipeline import ReZroSparseTopkGPTPipe
from transformers import BertTokenizer
from ds_util import get_argument_parser
from transformers import get_cosine_schedule_with_warmup
from deepspeed.pipe import PipelineModule
import wandb
import os
import logging
from itertools import cycle


logger = logging.getLogger("ReZeroSparsetopkGPTX")
log_formatter = logging.Formatter(
"%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def pretrain():
    """Main Train
    1) setup model, optimizer and lr_schedule
    2) set dataset
    3) train the model
    """
    args = get_arguments()
    logger.info('set seed')
    torch.manual_seed(9)
    deepspeed.runtime.utils.set_random_seed(9)

    logger.info('initialize NCCL & CUDA')
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank) # local rank passed from distributed launcher

    config = ModelConfig(config_path=args.config).get_config()

    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

    logger.info('load gptx dataset')
    # dataset = GPTXDatasetV2(tokenizer, config.max_seq_len, config.data_path)
    dataset = gptx_datset(config, tokenizer, GPTXDatasetV2)

    wandb.init(project="rezero_sparsetopk_gpt")

    train_dataloader, eval_dataloader = build_dataloaders(config, dataset, train_test_split=0.1)
    logger.info(f'train data length: {len(train_dataloader)}')
    logger.info(f'eval data length : {len(eval_dataloader)}')

    config.max_train_step = len(train_dataloader) * config.epoch
    config.max_eval_step = len(eval_dataloader)

    logger.info('set GPTX model, optimizer, scheduler')
    model, optimizer, lr_scheduler = setup_model_and_optimizer(config)

    logger.info('deepspeed initialize')
    model,optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        dist_init_required=False)

    logger.info('start training')
    train(config=config,
          model=model,
          train_dataloader=cycle(train_dataloader))

    evaluate(config, model, eval_dataloader=cycle(eval_dataloader))
    
def cross_entropy(lm_logits, labels):
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss
def train(config,
          model,
          train_dataloader):

    # Set train mode
    model.train()
    
    logger.info('start pretrainig iteration')
    
    for i in range(config.max_train_step):
        loss = model.train_batch(data_iter=train_dataloader)
        wandb.log({'train': {'loss': loss.item(), 'perplexity': torch.exp(loss)}})

    train_result = {"loss": loss.item(), "ppl": torch.exp(loss)}

    return train_result

def evaluate(config, model, eval_dataloader):
    model.eval()

    with torch.no_grad():
        for _ in range(config.max_eval_step):
            loss = model.eval_batch(data_iter=eval_dataloader, return_logits=False)
            wandb.log({'eval': {'loss': loss.item(), 'perplexity': torch.exp(loss)}})

    eval_result = {"loss": loss.item(), "ppl": torch.exp(loss)}
    model.train()
    return eval_result

def setup_model_and_optimizer(config):
    """"""
    model = get_model(config)
    optimizer, model_params = get_optimizer(config, model)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, config=config)

    return model, optimizer, lr_scheduler

def get_model(config):
    model = ReZroSparseTopkGPTPipe(vocab_size= config.vocab_size,
                     dim = config.dim,
                     depth = config.depth,
                     n_head= config.n_head,
                     max_seq_len= config.max_seq_len)
    model = model.to_layer()
    model = PipelineModule(layers=model,
                           loss_fn=cross_entropy,
                           num_stages=config.num_stages)
    return model

def get_model_params( model):
    named_parameter = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_parameter if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in named_parameter if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

def get_optimizer(config, model):
    model_params = get_model_params(model)
    if config.optimizer['type']=='cpu_adam':
        # from deepspeed.ops.adam import DeepSpeedCPUAdam
        # optimizer = DeepSpeedCPUAdam(model_params,
        #                             **config.optimizer['params'])
        cpu_adam_optimizer = torch.optim.Adam
        optimizer = cpu_adam_optimizer(model_params,
                                    **config.optimizer['params'])
    elif config.optimizer['type']=='adam':
        from deepspeed.ops.adam import FusedAdam as Adam
        optimizer = Adam(model_params,
                         **config.optimizer['params'])
    return optimizer,model_params

def get_learning_rate_scheduler(optimizer, config):
    num_iter = config.max_train_step
    warmup_num_iter= num_iter * config.warmup_iter
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=warmup_num_iter,
                                                   num_training_steps=num_iter)
    return lr_scheduler

def build_dataloaders(config, dataset, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
    dataset_len = len(dataset)
    eval_len = int(dataset_len * train_test_split)
    train_len = dataset_len - eval_len
    train_dataset, eval_dataset = random_split(dataset, (train_len, eval_len))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=train_shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=eval_shuffle)
    logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                     eval_dataloader  size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

    # return iter(train_loader), iter(eval_loader)

    return train_loader, eval_loader

def gptx_datset(config, tokenizer, dataset_obj):
  cache_data_path = f'{config.cache_path}/{config.model_name}.pickle'
  cache_dir_path= os.path.dirname(cache_data_path)

  if os.path.exists(cache_data_path): # 캐시 데이터가 존재하는 경우
    dataset = torch.load(cache_data_path)
    return dataset
  else: # 캐시 데이터가 없는 경우
    if not os.path.exists(cache_dir_path):
      os.makedirs(cache_dir_path) # 캐시 디렉토리 경로 생성

    dataset = dataset_obj(tokenizer, config.max_seq_len, config.data_path)
    torch.save(dataset, cache_data_path) # 데이터 저장

    return dataset

def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # no cuda mode is not supported
    args.no_cuda = False

    return args

if __name__ == '__main__':
    pretrain()
