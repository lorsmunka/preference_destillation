import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def create_evaluation_prompt(sentence):
    prompt = f"""Analyze the following sentence and return a JSON object with these evaluations:

    Sentence: "{sentence}"

    Return JSON format:
    {{
    "tone": "aggressive | rude | neutral | polite | friendly",
    "sentiment": "negative | neutral | positive",
    "safety": "harmful | safe",
    "toxicity": "toxic | respectful"
    }}

    JSON:
    """
    return prompt


def log_inf_file(data, name=None):
    os.makedirs("./logs", exist_ok=True)

    time_stamp = str(int(time.time()))

    file_name = f"./logs/{name + '_' if name else ''}{time_stamp}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Logged to", file_name)


def tokenize_and_log():
    all_possible_evaluation_tokens = '{"tone": "aggressive rude neutral polite friendly", "sentiment": "negative neutral positive", "safety": "harmful safe", "toxicity": "toxic respectful" }'
    catalyst_tokens = 'the a is of and to in that it you'
    auxiliary_tokens = 'very quite somewhat extremely slightly moderately'
    structural_tokens = '{ } " : , [ ]'
    combined_tokens = all_possible_evaluation_tokens + " " + \
        catalyst_tokens + " " + auxiliary_tokens + " " + structural_tokens

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    tokens = tokenizer.tokenize(combined_tokens)
    unique_tokens = sorted(set(tokens))
    return unique_tokens


unique_tokens = tokenize_and_log()


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it", device_map="auto")
    return tokenizer, model


def prepare_inputs_for_gpu(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    return inputs


def get_next_token_probabilities(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
    last_token_logits = logits[0, -1, :]
    return last_token_logits.softmax(dim=-1)


def extract_logits_with_other_token(tokenizer, logits):
    token_names = []
    token_logits = []
    used_indices = set()

    for token_text in unique_tokens:
        token_ids = tokenizer.convert_tokens_to_ids([token_text])
        if token_ids[0] != tokenizer.unk_token_id:
            token_id = token_ids[0]
            token_logits.append(logits[token_id].item())
            token_names.append(token_text)
            used_indices.add(token_id)

    other_indices = [i for i in range(len(logits)) if i not in used_indices]
    if other_indices:
        other_logits = logits[other_indices]
        max_logit = other_logits.max()
        other_logit_sum = max_logit + \
            torch.log(torch.sum(torch.exp(other_logits - max_logit)))
        token_logits.append(other_logit_sum.item())
        token_names.append("__OTHER__")

    return token_names, token_logits


def sort_by_logits(token_names, token_logits):
    token_logit_pairs = list(zip(token_names, token_logits))
    token_logit_pairs.sort(key=lambda x: x[1], reverse=True)
    return {token: logit for token, logit in token_logit_pairs}


def get_top_k_probabilities(tokenizer, probabilities, top_k):
    top_probs, top_indices = probabilities.topk(top_k)
    result = {}
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = tokenizer.decode([idx])
        result[token] = prob
    return result


def get_probability(data):
    tokenizer, model = load_model_and_tokenizer()
    inputs = prepare_inputs_for_gpu(tokenizer, data)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    token_names, token_logits = extract_logits_with_other_token(
        tokenizer, logits)
    result = sort_by_logits(token_names, token_logits)

    return result


log_inf_file(get_probability(create_evaluation_prompt(
    "I hate pizza.")), "gemma_labelspace_probabilities")
