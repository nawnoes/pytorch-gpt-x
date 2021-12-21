from model.pipeline import ReZroSparseTopkGPTPipe
from common.arg import ModelConfig
from transformers import BertTokenizer
from model.transformer import LayerNorm

config = ModelConfig(config_path="./config_rezero_sparsetopk.json").get_config()

tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

model = ReZroSparseTopkGPTPipe(vocab_size= config.vocab_size,
                     dim = config.dim,
                     depth = config.depth,
                     n_head= config.n_head,
                     max_seq_len= config.max_seq_len)

weight_decay_params = {"params": [], 'weight_decay': config.weight_decay}
no_weight_decay_params = {"params": [], 'weight_decay': 0.0}
for module in model:
    if isinstance(module,LayerNorm) or config.weight_decay==0.0:
        no_weight_decay_params["params"].extend(
            [p for p in list(module._parameters.values()) if p is not None]
        )
    else:
        weight_decay_params["params"]\
            .extend([p for n, p in list(module._parameters.items()) if p is not None and n != "bias"])
        no_weight_decay_params["params"]\
            .extend([p for n, p in list(module._parameters.items()) if p is not None and n == "bias"])