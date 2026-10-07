import sys, torch
sys.path.insert(0, r"D:\preference_destillation")
torch.manual_seed(0)
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
t = cfg.text_config
t.hidden_size = 32; t.intermediate_size = 64; t.num_hidden_layers = 4; t.num_attention_heads = 2; t.num_key_value_heads = 1
t.head_dim = 16; t.global_head_dim = 16; t.sliding_window = 8; t.num_global_key_value_heads = 1
t.layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
for sub in ("vision_config", "audio_config"):
    c = getattr(cfg, sub, None)
    if c is not None:
        for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"):
            if hasattr(c, k):
                setattr(c, k, {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 16}[k])
model = AutoModelForCausalLM.from_config(cfg).float().eval()
print(type(model).__name__, sum(p.numel() for p in model.parameters()))
exec(open(r"C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_datagen\align_body.py", encoding="utf-8").read())
