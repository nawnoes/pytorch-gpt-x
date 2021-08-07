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
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
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
                                from_seq_len,
                                to_seq_len,
                                seed=None,
                                plan_from_length=None,
                                plan_num_rand_blocks=None,
                                output_attentions=None
                               ):
        """
        block sparse attention as suggested in BigBird paper
        ITC:
            global token: 2 x block_size
            window token: 3 x block_size
            random token: num_rand_tokens x block_size
        ETC:
            global token: extra_globals_tokens + 2 x block_size
            window token: 3 x block_size
            random token: num_rand_tokens x block_size

        *Note:
            1) Currently ETC is not supported
            2) Window size is fixed to 3 block & it can be changed only by changing `block_size`
            3) Number of global blocks are fixed (2 block) & global tokens can be controlled only by `block_size`.
        """

        # Shorthands
        h = num_attention_heads
        r = num_rand_blocks
        rsqrt_d = 1/math.sqrt(attention_head_size)
        b = batch_size
        m = from_seq_len
        n = to_seq_len
        wm = from_block_size
        wn = to_block_size

        # Generate random attention and corresponding masks
        np.random.seed(seed)
        if from_seq_len in [1024, 3072, 4096]: # old plans used in paper
            rand_attention = [self.]

        return None

    @staticmethod
    def _block_rand_mask(from_seq_len, to_seq_len, from_block_size, to_block_size, num_rand_blocks, last_idx = -1):
        assert(from_seq_len//from_block_size == to_seq_len // to_block_size), "Error the number of blocks needs to be same"

        rand_attention = np.zero((from_seq_len // from_block_size -2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_len // to_block_size -1, dtype=np.int32)

        last = to_seq_len // to_block_size -1

        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) -1

        r = num_rand_blocks # shorthand
        for i in range(1, from_seq_len// from_block_size -1):
            start = i - 2
            end = i

            if i == 1:
                rand_attention[i -1, : ] = np.random.permutation(middle_seq[2:last])[:r]
            elif i ==2:
                rand_attention[i-1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i== from_seq_len // from_block_size -3:
                rand_attention[i-2,:] = np.random.permutation(middle_seq[:last])[:r]
            elif i==from_seq_len//from_block_size-2:
                rand_attention[i-1,:] = np.random.permutation(middle_seq[:last])[:r]
            else:
                if start>last:
                    start = last
                    rand_attention[i-1,:]=np.random.permutation(middle_seq[:start])[:r]
                elif (end +1) == last:
                    rand_attention[i-1,:]=np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attention[i-1,:]=np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end+1:last]))
                    )[:r]
        return rand_attention


