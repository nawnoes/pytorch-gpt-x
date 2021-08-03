"""
Rewriting BigBird Block sparse attention
Reference
github: vasudevgupta7/transformers
url: https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


class BlockSparseAttention(nn.Module):
    def __init__(self, config, seed = None):
        super().__init__()
        self.max_seq_leg = config.max_seq_leg
        self.seed = seed

        self.num_attention_heads = config.n_head
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.dim/config.n_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.dim, config.)
    def transpose_for_scores(self, x):
        pass
    
    def forward(self):
        pass