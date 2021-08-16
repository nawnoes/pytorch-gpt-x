# Pytorch GPT-X
My Own Pytorch GPT-X 

## Abstract
Train GPT-3 model on V100(16GB Mem) Using Reformer, Transformer. 

## Model
### Transformer
## Train
### DeepSpeed
## TODO

- [x] ~~ReZero~~
- [ ] Performer
- [ ] Residual Attention
- [ ] Shifted Tokens
- [ ] Macaron architectures
- [ ] Explicit Sparse Transformer
- [ ] Deepspeed train on single GPU
- [ ] Deepspeed parallel trainig on 2 V100 GPU with 16GB Memory

## Parameter For Few-shot
The 175B parameter model is very large, but a large model is needed for Few-Shot Learning.
So this repository try to use DeepSpeed for training extremely big model.
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbcCkzC%2FbtqEzhJ441q%2FCr6nzgvZHP4cDBj6bksKf0%2Fimg.png)

## Config
|   model_name    |n_params | n_layer | d_model | n_heads | d_head | batch_size | learning_rate |
|:---------------:|---------|---------|---------|---------|--------|------------|---------------|
|   GPT-3 175B    |  175B   |    96   |  12288  |    96   |   128  |    3.2M    |   0.6 x 10^-4 |
|   GPT-3 13B     |  13B    |    40   |  5140   |    40   |   128  |     2M     |   1.0 x 10^-4 |
|   GPT-3 6.7B    |  6.7B   |    32   |  4096   |    32   |   128  |     2M     |   1.2 x 10^-4 |
|   GPT-3 2.7B    |  2.7B   |    32   |  25560  |    32   |   80   |     1M     |   1.6 x 10^-4 |

# References
### Transformer
- [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
  
### DeepSpeed
- [DeepSpeed](https://www.deepspeed.ai/)
- [DeepSpeed Core API Doc](https://deepspeed.readthedocs.io/en/latest/index.html)

### ReZero
- [/majumderb/rezero](https://github.com/majumderb/rezero/blob/master/rezero/transformer/rztx.py)