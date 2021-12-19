import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam

def self_attention(query, key, value, mask=None, causal=False):
  key_transpose = torch.transpose(key,-2,-1)                      # (bath, n_head, d_k, token_)
  matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
  d_k = query.size()[-1]
  attention_score = matmul_result/math.sqrt(d_k)                  # Scale

  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e20)
  # is Decoder
  if causal:
    query_len = query.size()[2]
    i, j = torch.triu_indices(query_len, query_len, 1)
    attention_score[:, :, i, j] = -1e4

  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  result = torch.matmul(softmax_attention_score,value)

  return result, softmax_attention_score


class MultiHeadAttention(nn.Module):
  def __init__(self, n_head =8 , d_model = 512,dropout = 0.1, causal=False):
    super(MultiHeadAttention,self).__init__()

    self.n_head = n_head
    self.d_model = d_model
    self.d_k = self.d_v = d_model // n_head
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

    query = self.w_q(query).view(batche_num, -1, self.n_head, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.n_head, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.n_head, self.d_k).transpose(1, 2)

    attention_result, attention_score = self.self_attention(query, key, value, mask, self.causal)
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.n_head * self.d_k)
    attn_output = self.w_o(attention_result)


    return attn_output

def explicit_sparse_attention(q, k ,v, mask=None, causal=None, sparse_topk=8):
    key_transose = torch.transpose(k,-2, -1)
    dot = torch.matmul(q, key_transose)

    head_dim = q.size()[-1]
    dot = dot / head_dim ** 0.5

    # Encoder input mask
    if mask:
      dot = dot.masked_fill(mask, 1e-4)

    # Causal Look-Ahead mask
    if causal:
      q_length = q.size()[2]
      i,j = torch.triu_indices(q_length, q_length,1)
      dot[:, :, i, j] = -1e4

    # Explicit sparse attention mask
    if sparse_topk > 0:
      topk, _ = dot.topk(sparse_topk,dim=-1)
      vk = topk[...,-1].unsqueeze(-1).expand_as(dot)
      explicit_sparse_mask = dot < vk # vk보다 낮은값들이 true
      dot.masked_fill(explicit_sparse_mask, -1e4)

    attn_score = F.softmax(dot,dim=-1)
    attn_value = torch.matmul(attn_score, v)

    return attn_value, attn_score

class SparseTopkMultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_head, dropout=0.1, causal=False, sparse_topk=8):
    super().__init__()
    self.d_model=d_model
    self.k_head_dim = d_model//n_head
    self.v_head_dim = d_model//n_head

    self.n_head=n_head
    self.dropout=dropout
    self.causal=causal
    self.sparse_topk=sparse_topk

    self.w_query = nn.Linear(d_model, d_model)
    self.w_key = nn.Linear(d_model, d_model)
    self.w_value = nn.Linear(d_model, d_model)

    self.w_out = nn.Linear(d_model, d_model)
    self.self_attention = explicit_sparse_attention

    self.dropout=nn.Dropout(dropout)

  def forward(self, q, k, v, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)

    batch_size = q.size()[0]
    query = self.w_query(q).view(batch_size, -1, self.n_head, self.k_head_dim).transpose(1,2)
    key = self.w_key(k).view(batch_size, -1, self.n_head, self.k_head_dim).transpose(1,2)
    value = self.w_value(v).view(batch_size, -1, self.n_head, self.v_head_dim).transpose(1,2)

    attn_value, attn_score = self.self_attention(q=query, k=key, v=value, mask=mask,sparse_topk=self.sparse_topk)
    attn_value = attn_value.transpose(1,2).contiguous().view(batch_size, -1, self.n_head * self.k_head_dim)

    attn_value = self.w_out(attn_value)

    return self.dropout(attn_value)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)
        self.w_2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if activation == 'gelu':
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
  def __init__(self, d_model,n_head, dropout):
    super(Decoder,self).__init__()
    self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, n_head= n_head,causal=True)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward= FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)


  def forward(self, x):
    x = self.residual_1(x, lambda x: self.masked_multi_head_attention(x, x, x))
    x = self.residual_2(x, self.feed_forward)

    return x

class RZDecoder(nn.Module):
  def __init__(self, d_model,n_head, dropout):
    super().__init__()
    self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
    self.rezero_1 = ReZero(dropout)

    self.feed_forward = FeedForward(d_model)
    self.rezero_2 = ReZero(dropout)
  def forward(self, x):
    x_1 = self.masked_multi_head_attention(x,x,x)
    x = x + self.rezero_1(x_1)

    x_2 = self.feed_forward(x)
    x = x + self.rezero_2(x_2)

    return x

class ReZeroSparseTopkDecoder(nn.Module):
  def __init__(self, d_model,n_head, dropout, sparse_topk=8):
    super().__init__()
    self.masked_multi_head_attention = SparseTopkMultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout, causal=True, sparse_topk=sparse_topk)
    self.rezero_1 = ReZero(dropout)

    self.feed_forward = FeedForward(d_model)
    self.rezero_2 = ReZero(dropout)

  def forward(self, x):
    x_1 = self.masked_multi_head_attention(x,x,x)
    x = x + self.rezero_1(x_1)

    x_2 = self.feed_forward(x)
    x = x + self.rezero_2(x_2)

    return x

class ReZero(nn.Module):
  def __init__(self, dropout):
    super().__init__()
    self.res_weight = nn.Parameter(torch.Tensor([0]))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.dropout(x) * self.res_weight

    return x

class Embedding(nn.Module):
  def __init__(self, vocab_size, dim, max_seq_len):
    super().__init__()
    self.token_emb = nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim, max_seq_len)
  def forward(self, input_ids):
    x= self.token_emb(input_ids)
    x= x+self.position_emb(input_ids).type_as(x)
    return x

class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

class GPT2(nn.Module):
  def __init__(self,
               vocab_size,
               dim,
               depth,
               max_seq_len,
               n_head,
               dropout=0.1):
    super(GPT2, self).__init__()

    # Embedding
    self.embedding = Embedding(vocab_size, dim, max_seq_len)

    # Decoders
    self.decoders = nn.Sequential(*[Decoder(d_model=dim, n_head=n_head, dropout=dropout) for _ in range(depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, input_ids, labels):

    x = self.embedding(input_ids)
    x = self.decoders(x)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss

class ReZeroSparseTopkGPT(nn.Module):
  def __init__(self,
               vocab_size,
               dim,
               depth,
               max_seq_len,
               n_head,
               dropout=0.1):
    super(ReZeroSparseTopkGPT, self).__init__()

    # Embedding
    self.embedding = Embedding(vocab_size, dim, max_seq_len)

    # Decoders
    self.decoders = nn.Sequential(*[ReZeroSparseTopkDecoder(d_model=dim, n_head=n_head, dropout=dropout) for _ in range(depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, input_ids, labels):

    x = self.embedding(input_ids)
    x = self.decoders(x)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss

class LitGPT2(pl.LightningModule):
  def __init__(self,
               vocab_size,
               dim,
               depth,
               max_seq_len,
               head_num,
               dropout=0.1):
    super(LitGPT2, self).__init__()

    # Embedding
    self.embedding = Embedding(vocab_size, dim, max_seq_len)

    # Decoders
    # self.decoders = nn.Sequential(*[Decoder(d_model=dim, n_head=head_num, dropout=dropout) for _ in range(depth)])
    # self.decoders = nn.Sequential(*[RZDecoder(d_model=dim, n_head=head_num, dropout=dropout) for _ in range(depth)])
    self.decoders = nn.Sequential(*[ReZeroSparseTopkDecoder(d_model=dim, n_head=head_num, dropout=dropout) for _ in range(depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def configure_optimizers(self):
    # DeepSpeedCPUAdam provides 5x, 7x speedup over torch.optim.adma(w)
    return DeepSpeedCPUAdam(model_params=self.parameters(),
                            lr=5e-4)
    # return FusedAdam(self.parameters())

  def forward(self, input_ids, labels):
    x = self.embedding(input_ids)
    x = self.decoders(x)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss

  def training_step(self, train_batch, batch_idx):
    input_ids, labels = train_batch
    lm_logits, loss = self.forward(input_ids, labels)
    perplexity = torch.exp(loss)

    self.log('train_loss', loss, prog_bar=True)
    self.log('train_ppl', perplexity, prog_bar=True)

    tb_logs = {'train_loss':loss, 'train_ppl':perplexity}

    return {'loss':loss, 'log': tb_logs}

  def validation_step(self, val_batch, batch_idx):
    input_ids, labels = val_batch
    lm_logits, loss = self.forward(input_ids, labels)
    perplexity = torch.exp(loss)

    self.log('val_loss', loss, prog_bar=True)
    self.log('val_ppl', perplexity, prog_bar=True)

    tb_logs = {'val_loss': loss, 'val_ppl': perplexity}

    return {'loss': loss, 'log': tb_logs}

  def validation_epoch_end(self, outputs):
    loss, avg_ppl = 0, 0
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    for x in outputs:
      avg_ppl += x['log']['val_ppl']
    avg_ppl/len(outputs)
    logs = {'avg_val_loss':avg_loss, 'avg_val_ppl':avg_ppl}

    return {'avg_val_loss': avg_loss, 'avg_val_ppl': avg_ppl, 'log': logs, 'progress_bar': logs}
