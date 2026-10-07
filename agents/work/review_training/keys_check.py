import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, r"D:\preference_destillation")
import torch
from library.model.transformer import Transformer
from library.shared import load_input_vocabulary
from library.domain import as_domain
for run in ["e01-reddit-pure-kl-s1", "e01-postgen-pure-kl-s1", "ideaA-vocab-aux000-s1"]:
    info = json.load(open(rf"D:\preference_destillation\runs\{run}\info.json"))
    model = Transformer(domain=as_domain(info["domain"]), teacher_model=info["teacher_model"], hidden_dim=info["hidden_dim"],
                        num_layers=info["num_layers"], num_heads=info["num_heads"], dropout=info["dropout"],
                        auxiliary_token_percentage=info.get("auxiliary_token_percentage", 1.0),
                        input_vocabulary=load_input_vocabulary(info["domain"], info["teacher_model"]))
    state = torch.load(rf"D:\preference_destillation\runs\{run}\checkpoints\checkpoint_epoch_{info['epoch_count']-1}.pt", weights_only=True, map_location="cpu")["model_state_dict"]
    print(run, "rotary cache saved len:", state["rotary_embedding.cos_cached"].shape, "longest:", info.get("longest_training_sequence"))
    state.pop("rotary_embedding.cos_cached"); state.pop("rotary_embedding.sin_cached")
    result = model.load_state_dict(state, strict=False)
    print("   missing:", result.missing_keys, "unexpected:", result.unexpected_keys)
