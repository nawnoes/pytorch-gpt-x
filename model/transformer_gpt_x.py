import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

"""
self-Attention의 경우 Query Q, Key K, Value V를 입력으로 받아
MatMul(Q,K) -> Scale -> Masking(opt. Decoder) -> Softmax -> MatMul(result, V)

"""

def self_attention(query, key, value, mask=None, causal=False):
  key_transpose = torch.transpose(key,-2,-1)                      # (bath, head_num, d_k, token_)
  matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
  d_k = query.size()[-1]
  attention_score = matmul_result/math.sqrt(d_k)                  # Scale

  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e4)

  if causal:
    query_len = query.size()[2]
    # causal_mask = torch.tril(torch.ones(query_len, query_len))
    # attention_score = attention_score.masked_fill_(causal_mask == 0, -1e4)
    i, j = torch.triu_indices(query_len, query_len, 1)
    attention_score[:, :, i, j] = -1e4

  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  result = torch.matmul(softmax_attention_score,value)

  return result, softmax_attention_score


"""
멀티헤드 어텐션
MultiHead(Q,K,V) = Concat(head_1,head_2,...head_n)W^O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
W^Q는 모델의 dimension x d_k
W^K는 모델의 dimension x d_k
W^V는 모델의 dimension x d_v
W^O는 d_v * head 갯수 x 모델 dimension
논문에서는 헤더의 갯수를 8개 사용
"""
class MultiHeadAttention(nn.Module):
  def __init__(self, head_num =8 , d_model = 512,dropout = 0.1, causal=False):
    super(MultiHeadAttention,self).__init__()

    # print(d_model % head_num)
    # assert d_model % head_num != 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num
    self.causal = causal

    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = self_attention
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask = None):
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(1)

    batche_num = query.size(0)

    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

    attention_result, attention_score = self.self_attention(query, key, value, mask, self.causal)

    # 원래의 모양으로 다시 변형해준다.
    # torch.continuos는 다음행과 열로 이동하기 위한 stride가 변형되어
    # 메모리 연속적으로 바꿔야 한다!
    # 참고 문서: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)


    return self.w_o(attention_result)

"""
Position-wise Feed-Forward Networks
FFN(x) = max(0,xW_1 + b_1)W_2+b2
입력과 출력은 모두 d_model의 dimension을 가지고
내부의 레이어는 d_model * 4의 dimension을 가진다.
"""
class FeedForward(nn.Module):
  def __init__(self,d_model, dropout = 0.1):
    super(FeedForward,self).__init__()
    self.w_1 = nn.Linear(d_model, d_model*4)
    self.w_2 = nn.Linear(d_model*4, d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
"""
Layer Normalization
: layer의 hidden unit들에 대해서 mean과 variance를 구한다. 
nn.Parameter는 모듈 파라미터로 여겨지는 텐서
"""
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


"""
Decoder 블록은 FeedForward 레이어와 MultiHead 어텐션, Masked Multihead 어텐션 레이어를 가진다.
MaskedMultiHeadAttention -> MultiHeadAttention(encoder-decoder attention) -> FeedForward
"""
class Decoder(nn.Module):
  def __init__(self, d_model,head_num, dropout):
    super(Decoder,self).__init__()
    self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num, causal=True)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward = FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model, dropout=dropout)


  def forward(self, target):
    # target, x, target_mask, input_mask
    x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x))
    x = self.residual_2(x, lambda x: self.feed_forward(x))

    return x

class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

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

    for decoder in self.decoders:
      x = decoder(x)

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
