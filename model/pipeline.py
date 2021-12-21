# from model.n_transformer import GPTX
from model.transformer import ReZeroSparseTopkGPT
import deepspeed
#
# class GPTXPipe(GPTX):
#     def to_layer(self):
#         layers=[
#             self.embedding,
#             *self.decoders,
#             self.norm,
#             self.lm_head
#         ]
#         return layers

class ReZroSparseTopkGPTPipe(ReZeroSparseTopkGPT):
    def to_layer(self):
        layers = [
            self.embedding,
            *self.decoders,
            self.norm,
            self.lm_head
        ]

        return layers

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.decoders(x)

        lm_logits = self.lm_head(x)

        return lm_logits
