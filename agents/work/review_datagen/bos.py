from transformers import AutoTokenizer
for name in ["google/gemma-3-4b-it", r"D:\preference_destillation\models\gemma-4-12B-it"]:
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    raw = tok("Solve this.")["input_ids"][:3]
    chat = tok.apply_chat_template([{"role": "user", "content": "Solve this."}], tokenize=False, add_generation_prompt=True)
    templated = tok(chat)["input_ids"][:3]
    print(name.split("\\")[-1], "raw prompt ->", tok.convert_ids_to_tokens(raw), "| chat template ->", tok.convert_ids_to_tokens(templated))
