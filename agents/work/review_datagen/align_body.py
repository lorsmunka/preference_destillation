from transformers import AutoTokenizer
from library.data_gen.model_handler import ModelHandler
from library.shared.vocabulary import build_vocabulary, get_response_tokens
from library.domain.base import Domain
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
class StubDomain(Domain):
    name = "stub_align"
    stop_token = "QQQZZZ"
    max_steps = 7
    example_tokens = None
    example_responses = ["Answer: 1234567890; the a of and to in"]
    prompt_tokens = []
    def teacher_prompt(self, text): return "Prompt: " + text
    def task_metric(self): return None

domain = StubDomain()
handler = ModelHandler.__new__(ModelHandler)
handler.model_name = "tiny"; handler.domain_handler = domain; handler.max_generation_steps = domain.max_steps
handler.load_in_4bit = False; handler.chat_template = True; handler.thinking = False
handler.tokenizer = tok; handler.model = model; handler.last_generated_text = ""
handler.end_token_ids = {1, 106}
handler.vocabulary = build_vocabulary(tok, domain); handler.response_tokens = get_response_tokens(tok, domain)
handler.finish_example = lambda text, generated, steps, thinking: ({"sentence": text, "model_response": generated, "steps": steps}, None)

token_list = handler.vocabulary["token_list"]
reduced_ids = torch.tensor([handler.vocabulary["token_to_id"][t] for t in token_list])

def reference_logits(full_ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([full_ids])).logits[0][:, reduced_ids]

texts = ["hi", "a much longer input sentence that makes the left padding obvious and crosses the window", "medium length text here"]
batched = handler.generate_training_examples(texts)
worst_b_vs_s, worst_s_vs_ref = 0.0, 0.0
for text, (example, _) in zip(texts, batched):
    single, _ = handler.generate_training_example(text)
    btoks = [s["token"] for s in example["steps"]]; stoks = [s["token"] for s in single["steps"]]
    print("tokens equal batched vs single:", btoks == stoks, len(btoks))
    for a, b in zip(example["steps"], single["steps"]):
        worst_b_vs_s = max(worst_b_vs_s, (torch.tensor(a["logits"]) - torch.tensor(b["logits"])).abs().max().item())
    prompt_ids = tok(handler.build_prompt(text))["input_ids"]
    gen_ids = tok.convert_tokens_to_ids(stoks)
    ref = reference_logits(prompt_ids + gen_ids)
    for j, step in enumerate(single["steps"]):
        worst_s_vs_ref = max(worst_s_vs_ref, (torch.tensor(step["logits"]) - ref[len(prompt_ids) - 1 + j]).abs().max().item())
print("max |batched - single|:", worst_b_vs_s, " max |single - unpadded reference|:", worst_s_vs_ref)

# forced scoring vs reference
targets = ["Answer: 12;", "the a of and to in Answer: 1234567890;", "x"]
scored = handler.score_training_examples(texts, targets)
worst = 0.0
for text, target, (example, _) in zip(texts, targets, scored):
    prompt_ids = tok(handler.build_prompt(text))["input_ids"]
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    ref = reference_logits(prompt_ids + target_ids)
    assert [s["token"] for s in example["steps"]] == tok.convert_ids_to_tokens(target_ids)
    for j, step in enumerate(example["steps"]):
        worst = max(worst, (torch.tensor(step["logits"]) - ref[len(prompt_ids) - 1 + j]).abs().max().item())
    # also: a shifted-by-one comparison should be clearly different
    shifted = (torch.tensor(example["steps"][0]["logits"]) - ref[len(prompt_ids)]).abs().max().item() if len(target_ids) > 1 else None
    print("row", repr(target), "steps", len(example["steps"]), "shifted diff (should be large):", shifted)
print("max |forced - unpadded reference|:", worst)
