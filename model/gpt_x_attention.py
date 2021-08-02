"""
Rewriting BigBird Block sparse attention
Reference
github: vasudevgupta7/transformers
url: https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BlockSparseAttention(nn.Module):
    def __init__(self, config, seed = None):
        super().__init__()
        
    def transpose_for_scores(self, x):
        pass
    
    def forward(self):
        pass