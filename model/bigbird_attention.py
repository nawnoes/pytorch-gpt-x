"""
BigBird Block Sparse Attention
github: vasudevgupta7/transformers
url: https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513
"""

import torch
import torch.nn.functional as F
import math
import numpy as np

def bigbird_block_sparse_attention(
    
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
    output_attentions=None,
):
    
    # BigBird block-sparse attention as suggested in paper
    
    # ITC:
    #     global tokens: 2 x block_size
    #     window tokens: 3 x block_size
    #     random tokens: num_rand_tokens x block_size
    
    # ETC:
    #     global tokens: extra_globals_tokens + 2 x block_size
    #     window tokens: 3 x block_size
    #     random tokens: num_rand_tokens x block_size
    
    # Note:
    #     1) Currently, ETC is not supported.
    #     2) Window size is fixed to 3 blocks & it can be changed only by
    #     changing `block_size`.
    #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
    #     controlled only by `block_size`.
    
    assert (
        from_seq_length // from_block_size == to_seq_length // to_block_size
    ), "Error the number of blocks needs to be same!"
    
    # Define shorthands
    h = num_attention_heads
    r = num_rand_blocks
    rsqrt_d = 1 / math.sqrt(attention_head_size)
    b = batch_size
    m = from_seq_length
    n = to_seq_length
    wm = from_block_size
    wn = to_block_size
    
    # generate random attention and corresponding masks
    np.random.seed(seed)
    if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper
        rand_attn = [
            self._bigbird_block_rand_mask(self.max_seqlen, self.max_seqlen, wm, wn, r, last_idx=1024)[
            : (from_seq_length // wm - 2)
            ]
            for _ in range(h)
        ]
    else:
        if plan_from_length is None:
            plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(from_seq_length, wm, r)
        
        rand_attn = self._bigbird_block_rand_mask_with_head(
            from_seq_length=from_seq_length,
            to_seq_length=to_seq_length,
            from_block_size=wm,
            to_block_size=wn,
            num_heads=h,
            plan_from_length=plan_from_length,
            plan_num_rand_blocks=plan_num_rand_blocks,
        )
    
    rand_attn = np.stack(rand_attn, axis=0)
    rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
    rand_attn.unsqueeze_(0)
    rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
    
    rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, h, r, b, m, wm)
    
    blocked_query_matrix = query_layer.view(b, h, m // wm, wm, -1)
    blocked_key_matrix = key_layer.view(b, h, n // wn, wn, -1)
    blocked_value_matrix = value_layer.view(b, h, n // wn, wn, -1)
    
    # preparing block for randn attn
    gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
    gathered_key = gathered_key.view(b, h, n // wn - 2, r * wn, -1)  # [b, h, n//wn-2, r, wn, -1]
    gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
    gathered_value = gathered_value.view(b, h, n // wn - 2, r * wn, -1)  # [b, h, n//wn-2, r, wn, -1]
    
    # 1st block is global q[0] x (k[0], k[1], k[2], k[3], k[4] .... )
    
    # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)
    
    first_product = first_product * rsqrt_d
    first_product += (1.0 - to_mask) * -10000.0
    first_attn_weights = F.softmax(first_product, dim=-1)  # [b, h, wm, n]
    
    # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
    first_context_layer.unsqueeze_(2)
    
    # q[1] x (sliding_keys, random_keys, global_keys)
    
    second_key_mat = torch.cat(
        [
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, 1],
            blocked_key_matrix[:, :, 2],
            blocked_key_matrix[:, :, -1],
            gathered_key[:, :, 0],
        ],
        dim=2,
    )  # [b, h, (4+r)*wn, -1]
    second_value_mat = torch.cat(
        [
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, 1],
            blocked_value_matrix[:, :, 2],
            blocked_value_matrix[:, :, -1],
            gathered_value[:, :, 0],
        ],
        dim=2,
    )  # [b, h, (4+r)*wn, -1]
    
    # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
    second_seq_pad = torch.cat(
        [
            to_mask[:, :, :, : 3 * wn],
            to_mask[:, :, :, -wn:],
            torch.ones(b, 1, 1, r * wn, device=first_context_layer.device, dtype=first_context_layer.dtype),
        ],
        dim=3,
    )
    second_rand_pad = torch.cat(
        [
            torch.ones(b, h, wm, 4 * wn, device=first_context_layer.device, dtype=first_context_layer.dtype),
            rand_mask[:, :, 0],
        ],
        dim=3,
    )
    second_product = second_product * rsqrt_d
    second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * -10000.0
    second_attn_weights = F.softmax(second_product, dim=-1)  # [b , h, wm, (4+r)*wn]
    
    # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)
    
    second_context_layer.unsqueeze_(2)
    
    # q[-2:2] x (sliding_keys, random_keys, global_keys)
    
    # initialize q,k,v->q,k,v[-2:2]
    exp_blocked_key_matrix = torch.cat(
        [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
    )  # [b, h, m//wm-4, 3*wn, -1]
    exp_blocked_value_matrix = torch.cat(
        [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
        dim=3,
    )  # [b, h, m//wm-4, 3*wn, -1]
    middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
    
    # sliding attention scores for q[-2:2]
    # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
    inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
    #     ==> [b, h, m//wm-4, wm, 3*wn]
    inner_band_product = inner_band_product * rsqrt_d
    
    # randn attention scores for q[-2:2]
    # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
    rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
    #     ==> [b, h, m//wm-4, wm, r*wn]
    rand_band_product = rand_band_product * rsqrt_d
    
    # 1st block is global
    first_band_product = torch.einsum(
        "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    first_band_product = first_band_product * rsqrt_d
    
    # last block is global
    last_band_product = torch.einsum(
        "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    last_band_product = last_band_product * rsqrt_d
    
    # masking padded tokens
    inner_band_product += (1.0 - band_mask) * -10000.0
    first_band_product += (1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
    last_band_product += (1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0
    rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
    
    # completing attention scores matrix for all q[-2:2]
    band_product = torch.cat(
        [first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1
    )  # [b, h, m//wm-4, wm, (5+r)*wn]
    
    # safely doing softmax since attention matrix is completed
    attn_weights = F.softmax(band_product, dim=-1)  # [b, h, m//wm-4, wm, (5+r)*wn]
    
    # contibution of sliding keys
    # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
    context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, wn : 4 * wn], exp_blocked_value_matrix, ndim=5)
    #     ==> [b, h, m//wm-4, wm, -1]
    
    # adding contribution of random keys
    # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
    context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 4 * wn : -wn], gathered_value[:, :, 1:-1], ndim=5)
    #     ==> [b, h, m//wm-4, wm, -1]
    
    # adding contribution of global keys
    context_layer += torch.einsum(
        "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :wn], blocked_value_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
    context_layer += torch.einsum(
        "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -wn:], blocked_value_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
    
    # q[-2] x (sliding_keys, random_keys, global_keys)
    
    second_last_key_mat = torch.cat(
        [
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, -3],
            blocked_key_matrix[:, :, -2],
            blocked_key_matrix[:, :, -1],
            gathered_key[:, :, -1],
        ],
        dim=2,
    )  # [b, h, (4+r)*wn, -1]
    second_last_value_mat = torch.cat(
        [
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, -3],
            blocked_value_matrix[:, :, -2],
            blocked_value_matrix[:, :, -1],
            gathered_value[:, :, -1],
        ],
        dim=2,
    )  # [b, h, (4+r)*wn, -1]
    
    # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
    second_last_seq_pad = torch.cat(
        [
            to_mask[:, :, :, :wn],
            to_mask[:, :, :, -3 * wn :],
            torch.ones([b, 1, 1, r * wn], device=context_layer.device, dtype=context_layer.dtype),
        ],
        dim=3,
    )
    second_last_rand_pad = torch.cat(
        [
            torch.ones([b, h, wm, 4 * wn], device=context_layer.device, dtype=context_layer.dtype),
            rand_mask[:, :, -1],
        ],
        dim=3,
    )
    second_last_product = second_last_product * rsqrt_d
    second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
    second_last_attn_weights = F.softmax(second_last_product, dim=-1)  # [b, h, wm, (4+r)*wn]
    
    # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
    second_last_context_layer.unsqueeze_(2)
    
    # last block is global q[-1] x (k[0], k[1], k[2], k[3], .... )
    
    # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
    last_product = last_product * rsqrt_d
    last_product += (1.0 - to_mask) * -10000.0
    last_attn_weights = F.softmax(last_product, dim=-1)  # [b, h, wm, n]
    
    # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
    last_context_layer.unsqueeze_(2)
    context_layer = torch.cat(
        [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
        dim=2,
    )
    context_layer = context_layer.view((b, h, m, -1)) * from_mask
    context_layer = torch.transpose(context_layer, 1, 2)
    
    if output_attentions:
        # TODO(PVP): need to verify if below code is correct
        attention_probs = torch.zeros(b, h, m, n, dtype=torch.float, device=context_layer.device)
        
        # corresponding to `first_context_layer`
        attention_probs[:, :, :wm, :] = first_attn_weights
        
        # corresponding to `second_context_layer`
        attention_probs[:, :, wm : 2 * wm, : 3 * wn] = second_attn_weights[:, :, :, : 3 * wn]
        attention_probs[:, :, wm : 2 * wm, -wn:] = second_attn_weights[:, :, :, 3 * wn : 4 * wn]
        for p1, i1, w1 in zip(range(b), rand_attn, second_attn_weights):
            for p2, i2, w2 in zip(range(h), i1, w1):
                attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, 1, :, i2[0]] = w2[:, 4 * wn :].view(
                    wm, r, wn
                )
        
        # corresponding to `context_layer`
        for q_idx in range(m // wm - 4):
            slice = attention_probs.view(b, h, m // wm, wm, n // wn, wn)[:, :, 2:-2, :, 1:-1, :]
            slice[:, :, q_idx, :, q_idx : q_idx + 3, :] = attn_weights[:, :, q_idx, :, wn : 4 * wn].view(
                b, h, wm, 3, wn
            )  # inner_band_product
        attention_probs[:, :, 2 * wm : -2 * wm, :wn] = attn_weights[:, :, :, :, :wn].view(
            b, h, -1, wn
        )  # first_band_product
        attention_probs[:, :, 2 * wm : -2 * wm, -wn:] = attn_weights[:, :, :, :, -wn:].view(
            b, h, -1, wn
        )  # last_band_product
        for p1, i1, w1 in zip(range(b), rand_attn, attn_weights):
            for p2, i2, w2 in zip(range(h), i1, w1):
                for q_idx in range(1, len(i2) - 1):
                    attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, q_idx + 1, :, i2[q_idx]] = w2[
                                                                                                            q_idx - 1, :, 4 * wn : -wn
                                                                                                            ].view(wm, r, wn)
        
        # corresponding to `second_last_context_layer`
        attention_probs[:, :, -2 * wm : -wm, :wn] = second_last_attn_weights[:, :, :, :wn]
        attention_probs[:, :, -2 * wm : -wm, -3 * wn :] = second_last_attn_weights[:, :, :, wn : 4 * wn]
        for p1, i1, w1 in zip(range(b), rand_attn, second_last_attn_weights):
            for p2, i2, w2 in zip(range(h), i1, w1):
                attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, -2, :, i2[-1]] = w2[:, 4 * wn :].view(
                    wm, r, wn
                )
        
        # corresponding to `last_context_layer`
        attention_probs[:, :, -wm:, :] = last_attn_weights
    
    else:
        attention_probs = None
    
    return context_layer, attention_probs