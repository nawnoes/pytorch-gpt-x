from model.transformer import GPTX
import deepspeed

class GPTXPipe(GPTX):
    def to_layer(self):
        layers=[
            self.embedding,
            *self.decoders,
            self.norm,
            self.lm_head
        ]
        return layers
