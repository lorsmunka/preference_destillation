"""Can the trial's student (128 x 4 layers x 4 heads, fresh weights) learn to copy the
numbers of the newest message into its reply? Toy task on real conversation histories:
  history (a real slice conversation, numbers included) + <message>Mira has 26 basil and 30 mint. ...</message>
  -> <solution>\nbasil = 26\nmint = 30\ntotal = 26 + 30 = 56\nAnswer: 56\n</solution>
2 x 2, same budget (10 x 1,345 examples, cross-entropy only):
  updates: per example (the Trainer today: lr 0.0015, one optimizer step per example)
           vs batches of 32 (one step per 32, mean loss)
  data:    a fixed pool of 1,345 examples, 10 passes vs fresh examples every time
Reported on 300 fresh examples: copy digits (the two numbers) with the right message vs another
message's numbers (same history), and the answer digits (the reply's own sum, repeated).
Result 2026-10-14 (seed 1, CPU, ~10 min per arm), 1,345-example pool x10:
  batch-32:     copy 99.9% (other message 10%), answer 99.7%
  per example:  copy 45.2% (other message 11%), answer 42.3%
(the two fresh-data arms were not run: RAM.)

    python experiments/copy_check.py   # all four arms, about 45 min on the CPU"""
import json, math, random, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from library import RunStore
from library.model.student import StudentModel
from library.domain.math_tutor.slice import SliceDomain
from library.shared import PROMPT_DELIMITER

torch.manual_seed(0)
domain = SliceDomain()
store = RunStore()
run = store.get("slice-kd-s1")
_, path = store.resolve_checkpoint(run.name, "latest")
template = StudentModel.from_run(run, path, device="cpu", domain=domain)
tokenizer = template.model.tokenizer
token_to_index = {token: index for index, token in enumerate(template.model.vocabulary["token_list"])}
records = [json.loads(line) for line in open(domain.corpus_path, encoding="utf-8")]
histories = [r["text"][:r["text"].rfind("<message>")] for r in records if r["split"] == "train" and "<solution>" in r["text"]]

CANDIDATES = ["apples", "pears", "basil", "mint", "stickers", "marbles", "shells", "books", "cookies", "pencils", "cards",
              "eggs", "plates", "cups", "chairs", "boxes", "bags", "bottles", "coins", "stamps", "beads", "buttons", "toys", "balls"]
def fits(word):
    line_start = tokenizer.tokenize("\n" + word)[1:]
    mid = tokenizer.tokenize(" " + word)
    return all(t in token_to_index for t in line_start) and all(i in template.model.input_token_mapping for i in tokenizer.convert_tokens_to_ids(mid))
OBJECTS = [w for w in CANDIDATES if fits(w)]
NAMES = ["Mira", "Tom", "Ines", "Leo", "Ana", "Sam"]
print("objects:", OBJECTS, "| histories:", len(histories), flush=True)

def make(rng, numbers=None):
    a, b = numbers or (rng.randint(2, 99), rng.randint(2, 99))
    w1, w2 = rng.sample(OBJECTS, 2)
    name = rng.choice(NAMES)
    history = rng.choice(histories)
    message = f"<message>{name} has {a} {w1} and {b} {w2}. How many does {name} have in all?</message>"
    target = f"<solution>\n{w1} = {a}\n{w2} = {b}\ntotal = {a} + {b} = {a + b}\nAnswer: {a + b}\n</solution>"
    return {"history": history, "message": message, "target": target, "a": a, "b": b}

def encode(example, message=None):
    context = example["history"] + (message or example["message"])
    sentence = tokenizer.encode(context + PROMPT_DELIMITER, add_special_tokens=False)
    target_tokens = tokenizer.tokenize(example["target"])
    targets = [token_to_index[t] for t in target_tokens]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    inputs = template.model.remap_input_tokens(sentence + target_ids[:-1])
    # which target positions are the copied digits (lines 1-2) and the answer digits
    kinds, line = [], 0
    for t in target_tokens:
        if t == "\n":
            line += 1
        digit = t.strip("▁").isdigit()
        kinds.append("copy" if digit and line in (1, 2) else "answer" if digit and line == 4 else "other")
    return inputs, len(sentence), targets, kinds

def fresh_model():
    model = StudentModel.from_run(run, path, device="cpu", domain=domain).model
    model._init_weights()
    return model.train()

def lr_at(step, total, peak=0.0015, warmup_ratio=0.1):
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

def hidden_states(model, input_ids):
    """Transformer.forward without the output projection: only the reply positions get projected
    onto the 18,772 output tokens (all positions would need GBs of RAM on the CPU)."""
    x = model.dropout(model.input_embedding(input_ids))
    cos, sin = model.rotary_embedding(input_ids.shape[1])
    mask = model.create_causal_mask(input_ids.shape[1], input_ids.device)
    for layer in model.transformer_layers:
        x = layer(x, mask, cos[None, None], sin[None, None], None)
    return model.layer_norm(x)


def reply_logits(model, encoded):
    length = max(len(inputs) for inputs, _, _, _ in encoded)
    tensor = torch.zeros(len(encoded), length, dtype=torch.long)
    for row, (inputs, _, _, _) in enumerate(encoded):
        tensor[row, :len(inputs)] = torch.tensor(inputs)
    hidden = hidden_states(model, tensor)
    return [model.output_projection(hidden[row, start - 1:start - 1 + len(targets)])
            for row, (_, start, targets, _) in enumerate(encoded)]


def batch_loss(model, examples):
    encoded = [encode(e) for e in examples]
    losses = [F.cross_entropy(logits, torch.tensor(targets), reduction="sum")
              for logits, (_, _, targets, _) in zip(reply_logits(model, encoded), encoded)]
    return torch.stack(losses).sum(), sum(len(t) for _, _, t, _ in encoded)


def train(per_example, fixed_pool, budget=13450, seed=1):
    rng = random.Random(seed)
    model = fresh_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015)
    pool = [make(rng) for _ in range(1345)] if fixed_pool else None
    stream = [pool[i % 1345] for i in range(budget)] if fixed_pool else [make(rng) for _ in range(budget)]
    if fixed_pool:   # shuffle within each pass, like shuffle_seed
        stream = [e for p in range(10) for e in rng.sample(pool, len(pool))]
    group = 1 if per_example else 32
    updates = math.ceil(len(stream) / group)
    started = time.time()
    for step in range(updates):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, updates)
        optimizer.zero_grad(set_to_none=True)
        loss, tokens = batch_loss(model, stream[step * group:(step + 1) * group])
        (loss if per_example else loss / tokens).backward()   # the Trainer sums per example; batches: mean per token
        optimizer.step()
        if step % max(1, updates // 5) == 0:
            print(f"  update {step}/{updates}: loss/token {loss.item() / tokens:.3f} ({time.time() - started:.0f} s)", flush=True)
    return model

def evaluate(model, seed=99, count=300):
    model.eval()
    rng = random.Random(seed)
    tests = [make(rng) for _ in range(count)]
    results = {}
    for name in ("right", "other"):
        hit = {"copy": [0, 0], "answer": [0, 0], "all": [0, 0]}
        for example in tests:
            message = example["message"]
            if name == "other":   # same objects and name, different numbers: only the numbers differ
                c, d = rng.randint(2, 99), rng.randint(2, 99)
                message = example["message"].replace(f"has {example['a']} ", f"has {c} ", 1).replace(f"and {example['b']} ", f"and {d} ", 1)
            inputs, start, targets, kinds = encode(example, message)
            with torch.no_grad():
                predicted = reply_logits(model, [(inputs, start, targets, kinds)])[0].argmax(-1).tolist()
            for p, t, k in zip(predicted, targets, kinds):
                for key in (k, "all"):
                    if key in hit:
                        hit[key][0] += p == t; hit[key][1] += 1
        results[name] = {k: round(v[0] / v[1], 3) for k, v in hit.items()}
    return results

check_model = fresh_model().eval()
check_encoded = encode(make(random.Random(5)))
with torch.no_grad():
    full = check_model(torch.tensor([check_encoded[0]]))[0, check_encoded[1] - 1:check_encoded[1] - 1 + len(check_encoded[2])]
    mine = reply_logits(check_model, [check_encoded])[0]
print("reply_logits matches Transformer.forward:", torch.allclose(full, mine, atol=1e-5), flush=True)

for per_example in (False, True):   # batches first: faster
    for fixed_pool in (True, False):
        label = f"{'per-example' if per_example else 'batch-32'} / {'1,345 pool x10' if fixed_pool else 'fresh'}"
        print(f"== {label}", flush=True)
        started = time.time()
        model = train(per_example, fixed_pool)
        print(f"RESULT {label}: {evaluate(model)}  ({time.time() - started:.0f} s)", flush=True)
