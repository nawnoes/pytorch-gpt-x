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

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)
    
    def forward(self,
                hidden_states,
                output_attentions = None,
                band_mask=None,
                from_mask=None,
                to_mask=None,
                from_blocked_mask=None,
                to_blocked_mask=None
                ):
        # 현재는 decoder에는 지원되지 않음.
        # 추가적인 개발 필요.
        batch_size,seq_len, _ = hidden_states.size()
        to_seq_len = from_seq_len = seq_len
        from_block_size = to_block_size = self.block_size

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        context_layer, attention_probs = self.block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_len,
            to_seq_len,
            seed = self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attention = output_attentions
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_len, -1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    @staticmethod
    def torch_bmm_nd(input_1, input_2, dim= None):
        """
        Fast nd matrix multiplication
        """
        return torch.bmm(input_1.reshape((-1,) + input_1.shape[-2:]), input_2.reshape((-1,) + input_2.shape[-2:])).view(
            input_1.shape[:dim-2] + (input_1.shape[dim-2], input_2.shape[dim-2])
        )

    @staticmethod
    def torch_bmm_nd_transpose(input_1, input_2, dim=None):
        """
        Fast nd matrix multiplication with transpose
        """
        return torch.bmm(input_1.reshape((-1,) + input_1.shape[-2:]), input_2.reshape((-1,) + input_2.shape[-2:]).transpose(1,2)
                         ).view(input_1.shape[:dim-2] + (input_1.shape[dim-2], input_2.shape[dim-2]))

    def block_sparse_attention(self,
                                query_layer,
                                key_layer,
                                value_layer,
                                band_mask,
                                from_mask,
                                to_mask,
                                from_blocked_mask,
                                to_blocked_mask,
                                num_attention_heads,
                                num_rand_blocks,
                                attention_head_size,
                                from_block_size,
                                to_block_size,
                                batch_size,
                                from_seq_length,
                                to_seq_length,
                                seed=None,
                                plan_from_length=None,
                                plan_num_rand_blocks=None,
                                output_attentions=None
                               ):
        """
        block sparse attention as suggested in BigBird paper

        """