# GPT-X
Implementation of autoregressive lanague model(like GPT-2,3) using improved Transformer and deepspeed pipeline parallelism. 

## Improved Transformer
Transformer used in this repository attempts to improve the transformer using the additional modules below.
| Name                        | Description                                                                                | Link                                           |
|-----------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------|
| Rezero                      | Rezero Is All You Need                                                                     | [link](https://arxiv.org/abs/2003.04887)       |
| Explicit Sparse Transformer | Concentrated Attention Through Explicit Selection                                          | [link]( https://arxiv.org/abs/1912.11637 )     |
| Macaron Architecture        | Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View | [link]( https://arxiv.org/pdf/1906.02762.pdf ) |
| RealFormer                  | Residual Attention                                                                         | [link]( https://arxiv.org/abs/2012.11747 )     |
| ALiBi Position Embedding    | effective relative positional encoding                                                     |                                                |

## Model Description
| model_name | n_params | n_layer | d_model | n_heads | vocab_size | max_seq_len | learning_rate |
|:----------:|----------|---------|---------|---------|------------|-------------|---------------|
|  GPT-X 1B  |   1B     | 20      | 2048    | 16      | 22000      | 1024        | 2.0 x 10^-4   |

## DeepSpeed
DeepSpeed is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.  
  
### Piepline Parallelism
You can train 1B GPT-X Model using deepspeed pipeline parallelism on 2 V100 GPU(16G).

#### GPU Usage
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:00:06.0 Off |                    0 |
| N/A   42C    P0    44W / 250W |  16076MiB / 16130MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  On   | 00000000:00:07.0 Off |                    0 |
| N/A   45C    P0   168W / 250W |  16060MiB / 16130MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     29525      C   /home/ubuntu/anaconda3/bin/python          16065MiB |
|    1     29528      C   /home/ubuntu/anaconda3/bin/python          16049MiB |
+-----------------------------------------------------------------------------+
```
#### Pipeline Parallelism Log
```
[2021-12-31 12:24:20,042] [INFO] [engine.py:93:__init__] CONFIG: micro_batches=4 micro_batch_size=1
[2021-12-31 12:24:20,094] [INFO] [engine.py:151:__init__] RANK=1 STAGE=1 LAYERS=12 [11, 23) STAGE_PARAMS=548560916 (548.561M) TOTAL_PARAMS=1099214888 (1099.215M) UNIQUE_PARAMS=1099214888 (1099.215M)
[2021-12-31 12:24:20,094] [INFO] [engine.py:151:__init__] RANK=0 STAGE=0 LAYERS=11 [0, 11) STAGE_PARAMS=550653972 (550.654M) TOTAL_PARAMS=1099214888 (1099.215M) UNIQUE_PARAMS=1099214888 (1099.215M)
```

```
[2021-12-31 12:24:08,793] [INFO] [module.py:365:_partition_layers] Partitioning pipeline stages with method parameters
stage=0 layers=11
     0: Embedding
     1: ReZeroSparseTopkDecoder
     2: ReZeroSparseTopkDecoder
     3: ReZeroSparseTopkDecoder
     4: ReZeroSparseTopkDecoder
     5: ReZeroSparseTopkDecoder
     6: ReZeroSparseTopkDecoder
     7: ReZeroSparseTopkDecoder
     8: ReZeroSparseTopkDecoder
     9: ReZeroSparseTopkDecoder
    10: ReZeroSparseTopkDecoder
stage=1 layers=12
    11: ReZeroSparseTopkDecoder
    12: ReZeroSparseTopkDecoder
    13: ReZeroSparseTopkDecoder
    14: ReZeroSparseTopkDecoder
    15: ReZeroSparseTopkDecoder
    16: ReZeroSparseTopkDecoder
    17: ReZeroSparseTopkDecoder
    18: ReZeroSparseTopkDecoder
    19: ReZeroSparseTopkDecoder
    20: ReZeroSparseTopkDecoder
    21: LayerNorm
    22: Linear
  loss: cross_entropy
```


## TODO

- [x] ~~ReZero~~
- [x] ~~RealFormer, Residual Attention~~ 
- [x] ~~Macaron architectures~~
- [x] ~~Macaron architectures - layer Scale 0.5~~
- [x] ~~Explicit Sparse Transformer~~
- [x] ~~torch lightning~~
- [x] ~~Deepspeed train on single GPU~~
- [x] apply wandb
- [x] Deepspeed pipeline parallel trainig on 2 V100 GPU with 16GB Memory

## Parameter For Few-shot
GPT-3 has a 175B parameter, and the size of the model is important for few-shot learning. In this repository, I try to pretrain language model as large as possible using 2 V100 GPUs.

## GPT-3 Config
| model_name | n_params | n_layer | d_model | n_heads | d_head | batch_size | learning_rate |
|:----------:|----------|---------|---------|---------|--------|------------|---------------|
| GPT-3 175B | 175B     | 96      | 12288   | 96      | 128    |    3.2M    | 0.6 x 10^-4   |
| GPT-3 13B  | 13B      | 40      | 5140    | 40      | 128    |     2M     | 1.0 x 10^-4   |
| GPT-3 6.7B | 6.7B     | 32      | 4096    | 32      | 128    |     2M     | 1.2 x 10^-4   |
| GPT-3 2.7B | 2.7B     | 32      | 2560    | 32      | 80     |     1M     | 1.6 x 10^-4   |
| GPT-3 1.3B | 1.3B     | 24      | 2048    | 24      | 128    |     1M     | 2.0 x 10^-4   |

## Issue
- `AttributeError: module 'deepspeed' has no attribute 'zero'`: reinstall deepspeed
- `userwarning: cuda initialization: the nvidia driver on your system is too old`: reinstall pytorch following by cuda version
    **my solution**-GPU V100, cuda 10.1  

    ```sh
  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```
- `can't find CUDA_HOME path`: reinstall cuda
# References
**Transformer**

- [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
  
**DeepSpeed**

- [DeepSpeed](https://www.deepspeed.ai/)
- [DeepSpeed Core API Doc](https://deepspeed.readthedocs.io/en/latest/index.html)

**ReZero**

- [/majumderb/rezero](https://github.com/majumderb/rezero/blob/master/rezero/transformer/rztx.py)

**Explicit Sparse Transformer**

- [x-transformer: explicit_sparse_transformer](https://github.com/lucidrains/x-transformers/blob/2badf9261cda03e1497b5db62274b045cd827086/x_transformers/x_transformers.py#L469)

**Macaron Architecrue**

- [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/pdf/1906.02762.pdf)

**RealFormer Residual Attention**
- [cloneofsimo/RealFormer-pytorch](https://github.com/cloneofsimo/RealFormer-pytorch/blob/main/models.py)

**DeepSpeed**
- [PyTorch lightning DeepSpeed](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed)

**Pipeline Parallelism**
- [DeepSpeed Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
- [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/1fed12e8b375b0c54902827e7140d8266dfccd59/pipeline_parallelism)
- [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/49e60fe7ad14f6991a7fa678d3a0c330d09b9ff4/megatron/training.py)