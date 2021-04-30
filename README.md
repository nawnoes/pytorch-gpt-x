# Reformer GPT-3
GPT-3 Using reformer-pytorch. 

## Abstract
Train GPT-3 175B model on V100(16GB Mem) using Reformer that is memory efficient model.

## Parameter For Few-shot
The 175B parameter model is very large, but a large model is needed for Few-Shot Learning.
So this repository try to use DeepSpeed for training extremely big model.
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbcCkzC%2FbtqEzhJ441q%2FCr6nzgvZHP4cDBj6bksKf0%2Fimg.png)

## Config
|   model_name    | token_num |n_params | n_layer | d_model | n_heads | d_head | batch_size | learning_rate |
|:---------------:|-----------|---------|---------|---------|---------|--------|------------|---------------|
|   GPT-3 175B    |    1024   |  175B   |    96   |  12288  |    96   |   128  |    3.2M    |   0.6 x 10^-4 |