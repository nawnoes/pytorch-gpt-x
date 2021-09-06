import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def self_attention(query, key, value, mask=None, causal=False, explicit_topk=None, prev_attn=None):
  key_transpose = torch.transpose(key,-2,-1)                      # (bath, head_num, d_k, token_)
  matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
  d_k = query.size()[-1]
  attention_score = matmul_result/math.sqrt(d_k)                  # Scale

  pre_softmax_attn = None
  if prev_attn:
    attention_score = attention_score + prev_attn
    pre_softmax_attn = attention_score

  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e4)

  # is Decoder
  if causal:
    query_len = query.size()[2]
    # causal_mask = torch.tril(torch.ones(query_len, query_len))
    # attention_score = attention_score.masked_fill_(causal_mask == 0, -1e4)
    i, j = torch.triu_indices(query_len, query_len, 1)
    attention_score[:, :, i, j] = -1e4

  # Explicit Sparse Transformer
  if explicit_topk and explicit_topk< attention_score.shape[-1]:
    top, _ = attention_score.topk(explicit_topk, dim=-1) # return value, indices
    vk = top[...,-1].unsqueeze(-1).expand_as(attention_score)
    mask = attention_score < vk
    attention_score.masked_fill_(mask,-1e4)
    del mask

  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  result = torch.matmul(softmax_attention_score,value)

  return result, softmax_attention_score, pre_softmax_attn

class MultiHeadAttention(nn.Module):
  def __init__(self, head_num =8 , d_model = 512,dropout = 0.1, causal=False, explicit_sparse_attn_topk=None, residual_attn=False):
    super(MultiHeadAttention,self).__init__()

    # print(d_model % head_num)
    # assert d_model % head_num != 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num
    self.causal = causal
    self.explicit_topk = explicit_sparse_attn_topk
    self.residual_attn = residual_attn # Residual Attention

    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = self_attention
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask = None, prev_attn=None):
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(1)

    batche_num = query.size(0)

    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

    attention_result, attention_score, pre_softmax_attn = self.self_attention(query, key, value, mask, self.causal, self.explicit_topk, prev_attn)
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)


    return self.w_o(attention_result), pre_softmax_attn

class Scale(nn.Module):
  def __init__(self, scale_value, fn):
    super().__init__()
    self.scale_value = scale_value
    self.fn = fn
  def forward(self, input):
    x = self.fn(input)
    return x * self.scale_value


class FeedForward(nn.Module):
  def __init__(self,d_model, dropout = 0.1, activation='gelu'):
    super(FeedForward,self).__init__()
    self.w_1 = nn.Linear(d_model, d_model*4)
    self.w_2 = nn.Linear(d_model*4, d_model)
    self.dropout = nn.Dropout(p=dropout)
    
    if activation =='gelu':
        self.activation = F.gelu
    elif activation == 'relu':
        self.activation = F.relu

  def forward(self, x):
    return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm,self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
  def forward(self, x):
    mean = x.mean(-1, keepdim =True) # 평균
    std = x.std(-1, keepdim=True)    # 표준편차

    return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
  def __init__(self, size, dropout):
    super(ResidualConnection,self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout((sublayer(self.norm(x))))

class Residual(nn.Module):
  def __init__(self, dropout):
    super(Residual,self).__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer_output):
    return x + self.dropout(sublayer_output)

class Decoder(nn.Module):
  def __init__(self, d_model,head_num, dropout, rezero_use = True, explicit_sparse_attn_topk=8, macaron_net_use = False, residual_attn=False):
    """
    d_model: model hidden dimension
    head_num: number of attention head
    dropout: dropout probablity
    rezero_use = True : ReZero use or not
    explicit_sparse_attn_topk=8: Explicit sparse attention top-k. The origin paper suggest topk = 8. keep only the top 8 values before attention (softmax)
    """
    super(Decoder,self).__init__()

    # Macaron Architecture
    self.macaron = macaron_net_use

    if self.macaron:
      self.macaron_net = Scale(0.5,FeedForward(d_model, d_model))

    self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num, causal=True, explicit_sparse_attn_topk=explicit_sparse_attn_topk, residual_attn=residual_attn)
    self.residual_1 = ReZero(dropout) if rezero_use else Residual(dropout=dropout)

    self.feed_forward = FeedForward(d_model)
    if self.macaron:
      self.feed_forward = Scale(0.5, self.feed_forward)
    self.residual_2 = ReZero(dropout) if rezero_use else Residual(dropout=dropout)


  def forward(self, x, prev_attn=None):
    if self.macaron:
      x = self.macaron_net(x)
    # target = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x))
    # target = self.residual_2(target, lambda x: self.feed_forward(x))

    x, pre_softmax_attn = self.masked_multi_head_attention(query=x, key=x, value=x, prev_attn=prev_attn)
    x = self.residual_1(x)
    x = self.feed_forward(x)
    x = self.residual_2(x)

    return x, pre_softmax_attn

class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)
  
  
class ReZero(nn.Module):
  def __init__(self, dropout):
      super().__init__()
      self.g = nn.Parameter(torch.zeros(1))
      self.dropout = nn.Dropout(dropout)
      
  def forward(self, x, sublayer):
      x = sublayer(x)
      x = x * self.g
      return x + self.dropout(x)

  
class TransformerGPTX(nn.Module):
  def __init__(self,
               vocab_size,
               dim,
               depth,
               max_seq_len,
               head_num,
               dropout=0.1):
    super(TransformerGPTX, self).__init__()

    # Embedding
    self.token_emb = nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim, max_seq_len)

    # Decoders
    self.decoders = nn.ModuleList([Decoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, input_ids, labels):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    pre_attn =None
    for decoder in self.decoders:
      x, pre_attn = decoder(x, pre_attn)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss


if __name__=="__main__":
  pass
