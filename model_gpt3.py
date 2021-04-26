import torch.nn as nn
from reformer_pytorch import ReformerLM
from torch.nn import CrossEntropyLoss


class ReformerGPT3(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads, causal=True):
        super().__init__()
        self.reformer = ReformerLM(
                num_tokens= num_tokens,
                dim= dim,
                depth= depth,
                heads= heads,
                max_seq_len= max_seq_len,
                causal= causal,           # auto-regressive 학습을 위한 설정
                return_embeddings=True    # reformer 임베딩을 받기 위한 설정
            )
        self.lm_head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self,input_ids=None,labels=None,**kwargs):
        reformer_outputs = self.reformer(input_ids,**kwargs)
        hidden_states = reformer_outputs

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return lm_logits,loss
