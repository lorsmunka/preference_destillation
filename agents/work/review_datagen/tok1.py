import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
print("plain Hi:", tok("Hi")["input_ids"], "add_bos?", getattr(tok, "add_bos_token", None))
print("bos-prefixed text:", tok("<bos>Hi")["input_ids"])
chat = tok.apply_chat_template([{"role":"user","content":"Hi there"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
for target in ["Answer: 15;", "24 - 9 = 15\nAnswer: 15;", "<solution>24 - 9 = 15\nAnswer: 15;", " Answer: 15;", "\nAnswer: 15;", "35-23=12\n4*6=24\nAnswer: 7;"]:
    joint = tok(chat + target)["input_ids"]
    sep = tok(chat)["input_ids"] + tok(target, add_special_tokens=False)["input_ids"]
    print(repr(target), joint == sep, tok.convert_ids_to_tokens(tok(target, add_special_tokens=False)["input_ids"]))
print([ (t, tok.convert_ids_to_tokens(tok(t, add_special_tokens=False)["input_ids"])) for t in ["<|channel>thought\n<channel|>", "thought\n<channel|>"]])
print("decode channel ids:", [tok.decode([i]) for i in [100, 45518, 107, 101]])
