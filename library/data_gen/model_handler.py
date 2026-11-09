from time import time
from typing import Dict, List, Tuple, Optional

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache, LogitsProcessor,
                          StoppingCriteriaList, StopStringCriteria)

from library.shared.vocabulary import (
    build_vocabulary,
    get_response_tokens,
    extract_logits_as_vector,
)
from library.shared.config import teacher_weights_path
from library.shared.device import get_device
from library.domain import as_domain


MAX_THINKING_TOKENS = 1024
# Gemma 4 sometimes opens an empty thought channel before answering, thinking or not, in
# either of these spellings; it is skipped, not recorded.
THOUGHT_SCAFFOLDS = ("thought\n<channel|>", "<|channel>thought\n<channel|>")


def continues_scaffold(prefix: str) -> bool:
    return any(scaffold.startswith(prefix) for scaffold in THOUGHT_SCAFFOLDS)


class ReducedLogitsRecorder(LogitsProcessor):
    """Keeps each step's logits over the reduced vocabulary only (one small CPU row per
    sequence per step) and passes the scores on unchanged, so greedy decoding is untouched."""

    def __init__(self, token_ids: List[int]):
        self.token_ids = token_ids
        self.rows: List[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.rows.append(scores[:, self.token_ids].float().cpu())
        return scores


class ModelHandler:
    def __init__(self, model_name: str, domain, load_in_4bit: bool = False,
                 chat_template: bool = False, thinking: bool = False):
        self.model_name = model_name
        self.domain_handler = as_domain(domain)  # accepts a Domain object or a name string
        self.max_generation_steps = self.domain_handler.max_steps
        self.load_in_4bit = load_in_4bit
        self.chat_template = chat_template or thinking
        self.thinking = thinking
        self.tokenizer = None
        self.model = None
        self.vocabulary: Optional[dict] = None
        self.response_tokens: Optional[set] = None
        self.thought_end_token_id: Optional[int] = None
        self.last_generated_text = ""   # kept even when the example is rejected, for diagnostics
        self.stop_criteria: Dict[tuple, StopStringCriteria] = {}   # built once per set of stop strings

        self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self):
        start_time = time()
        weights = teacher_weights_path(self.model_name)   # local re-sharded copy when one exists
        print(f"Loading tokenizer and model{' (4-bit NF4)' if self.load_in_4bit else ''} from {weights}...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
        if self.load_in_4bit:
            # bitsandbytes NF4 keeps the weights 4-bit at runtime (~7 GB for 12B). Note that
            # compressed-tensors checkpoints (e.g. *-qat-w4a16-ct) cannot be used instead: this
            # transformers stack unpacks them to bf16 on the first forward pass.
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(weights, **load_kwargs)
        self.end_token_ids = self.find_end_token_ids()
        if self.thinking:
            self.thought_end_token_id = self.find_thought_end_token()
        elapsed_time = time() - start_time
        print(
            f"Loaded tokenizer and model -> took {elapsed_time:.2f} seconds.\n")

        self.create_vocabulary()

    def find_end_token_ids(self) -> set:
        """EOS plus the end-of-turn token of whichever Gemma generation this is; generation
        stops on any of them (the domain's own stop token normally fires first)."""
        ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        vocabulary = self.tokenizer.get_vocab()
        for token in ("<turn|>", "<end_of_turn>"):
            if token in vocabulary:
                ids.add(vocabulary[token])
        return ids

    def find_thought_end_token(self) -> int:
        """The special token that closes Gemma 4's thinking channel. Fails loudly on a
        tokenizer that has no such token, rather than silently generating without it."""
        candidates = [token for token in self.tokenizer.get_vocab() if "channel" in token and "|" in token]
        closers = [token for token in candidates if token.startswith("<channel")]
        if not closers:
            raise ValueError(f"{self.model_name}: thinking=True but no thought-channel close token in the "
                             f"tokenizer (channel-like tokens: {candidates[:6]})")
        return self.tokenizer.convert_tokens_to_ids(closers[0])

    def create_vocabulary(self):
        start_time = time()
        print("Creating vocabulary...")

        self.vocabulary = build_vocabulary(self.tokenizer, self.domain_handler)
        self.response_tokens = set(get_response_tokens(
            self.tokenizer, self.domain_handler))

        elapsed_time = time() - start_time
        print(f"Created vocabulary -> took {elapsed_time:.2f} seconds.\n")

    def build_prompt(self, text: str) -> str:
        prompt = self.domain_handler.teacher_prompt(text)
        if not self.chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking)

    def prompt_token_ids(self, prompt: str) -> List[int]:
        """The prompt's token ids with exactly one BOS in front. A chat template writes its own
        <bos> into the text, and some tokenizers add one by themselves (Gemma 3 does, Gemma 4
        does not), so the text is tokenised without special tokens and a BOS put in front only
        when the text has none."""
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is not None and token_ids[:1] != [bos_token_id]:
            token_ids = [bos_token_id] + token_ids
        return token_ids

    def encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Input ids and attention mask for one or more prompts, left-padded to one length,
        on the model's device."""
        rows = [self.prompt_token_ids(prompt) for prompt in prompts]
        length = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        device = get_device()
        return {
            "input_ids": torch.tensor([[pad] * (length - len(row)) + row for row in rows], device=device),
            "attention_mask": torch.tensor([[0] * (length - len(row)) + [1] * len(row) for row in rows], device=device),
        }

    def stopping_criteria(self, stop_strings: Optional[List[str]]) -> Optional[StoppingCriteriaList]:
        """Ends generation once any of `stop_strings` has been generated. The criterion is built
        the first time a set of stop strings is used and reused afterwards: passing stop_strings
        to generate() rebuilds it on every call, which takes 0.8-1.8 s on Gemma 4 (transformers'
        own cache never hits). It also fires on tokens that merely contain a stop string, such
        as "');" for ";", exactly as the domain's is_stop reads the decoded text."""
        if not stop_strings:
            return None
        key = tuple(stop_strings)
        if key not in self.stop_criteria:
            self.stop_criteria[key] = StopStringCriteria(self.tokenizer, list(key))
        return StoppingCriteriaList([self.stop_criteria[key]])

    def generate_thinking(self, input_ids: torch.Tensor):
        """Let the teacher think (greedy, plain generate — no logits kept), stopping when the
        thought channel closes. Returns (sequence including the thought, thought text)."""
        with torch.no_grad():
            output = self.model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                max_new_tokens=MAX_THINKING_TOKENS, eos_token_id=self.thought_end_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = output[0, input_ids.shape[1]:]
        thought = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return output, thought

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        return self.domain_handler.is_stop(last_token_decoded, generated_text)

    def generate_training_example(self, text: str) -> Tuple[Optional[Dict], Optional[str]]:
        start_time = time()

        prompt = self.build_prompt(text)
        inputs = self.encode_prompts([prompt])

        steps = []
        generated_text = ""
        sequence = inputs['input_ids']
        thinking = ""
        if self.thinking:
            sequence, thinking = self.generate_thinking(sequence)

        # KV-cached decoding: the prompt (and thought) is fed once, then one token per step —
        # the same path model.generate() takes. Re-feeding the whole prefix every step, as the
        # old loop did, gave different (worse) logits on Gemma 4 through bitsandbytes.
        cache = DynamicCache()
        pending = sequence
        last_token_decoded = ""
        # Gemma 4 sometimes opens an empty thought channel before answering. Tokens that could
        # be its start are held back and dropped only if the whole preamble follows; otherwise
        # they are real output (a "<" that turns out to open "<solution>" must not be lost).
        scaffold_text, held = "", []
        while not self.is_stop(last_token_decoded, generated_text) and len(steps) <= self.max_generation_steps:
            token_repr, token_decoded, logit_vector, predicted_token_index, next_token_id = self.generate_single_step(
                pending, cache, total_length=sequence.shape[1])
            if next_token_id in self.end_token_ids:
                break   # end of sequence / end of turn: not part of the response
            pending = torch.tensor([[next_token_id]], device=sequence.device)
            sequence = torch.cat([sequence, pending], dim=1)
            step = ({"token": token_repr, "logits": logit_vector, "predicted_token_index": predicted_token_index}, token_decoded)

            if not steps and scaffold_text is not None and continues_scaffold(scaffold_text + token_decoded):
                scaffold_text += token_decoded
                held.append(step)
                if scaffold_text in THOUGHT_SCAFFOLDS:   # the whole preamble: drop it
                    scaffold_text, held = None, []
                continue
            scaffold_text = None
            for held_step, held_text in held + [step]:
                steps.append(held_step)
                generated_text += held_text
            held = []
            last_token_decoded = token_decoded
        for held_step, held_text in held:   # output ended inside what looked like a preamble
            steps.append(held_step)
            generated_text += held_text

        elapsed_time = time() - start_time
        print(f"\tGenerated training example length of {len(steps)} tokens -> took {elapsed_time:.2f} seconds.")
        return self.finish_example(text, generated_text, steps, thinking)

    def generate_training_examples(self, texts: List[str]) -> List[Tuple[Optional[Dict], Optional[str]]]:
        """Several examples in one batched greedy model.generate call (left padding), for
        throughput. Same prompt, stop, scaffold and vocabulary rules as
        generate_training_example; the per-step logits come from ReducedLogitsRecorder.
        Thinking mode is not batched."""
        if self.thinking:
            raise ValueError("batched generation does not support thinking mode")
        start_time = time()
        inputs = self.encode_prompts([self.build_prompt(text) for text in texts])
        token_list = self.vocabulary["token_list"]
        recorder = ReducedLogitsRecorder([self.vocabulary["token_to_id"][token] for token in token_list])
        stop_strings = getattr(self.domain_handler, "stop_strings", None) or [self.domain_handler.stop_token]
        with torch.no_grad():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_generation_steps + 8,
                logits_processor=[recorder], stopping_criteria=self.stopping_criteria(stop_strings),
                eos_token_id=sorted(self.end_token_ids),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = output[:, inputs["input_ids"].shape[1]:].tolist()

        results, self.last_generated_texts = [], []
        for row_index, text in enumerate(texts):
            steps, generated_text, last_token_decoded = [], "", ""
            scaffold_text, held = "", []   # possible empty-thought preamble, held back (see generate_training_example)
            for position, token_id in enumerate(new_tokens[row_index]):
                if self.is_stop(last_token_decoded, generated_text) or len(steps) > self.max_generation_steps:
                    break
                if token_id in self.end_token_ids:
                    break
                token_decoded = self.tokenizer.decode([token_id])
                logit_vector = recorder.rows[position][row_index].tolist()
                step = ({"token": self.tokenizer.convert_ids_to_tokens([token_id])[0], "logits": logit_vector,
                         "predicted_token_index": int(max(range(len(logit_vector)), key=logit_vector.__getitem__))},
                        token_decoded)
                if not steps and scaffold_text is not None and continues_scaffold(scaffold_text + token_decoded):
                    scaffold_text += token_decoded
                    held.append(step)
                    if scaffold_text in THOUGHT_SCAFFOLDS:
                        scaffold_text, held = None, []
                    continue
                scaffold_text = None
                for held_step, held_text in held + [step]:
                    steps.append(held_step)
                    generated_text += held_text
                held = []
                last_token_decoded = token_decoded
            for held_step, held_text in held:
                steps.append(held_step)
                generated_text += held_text
            self.last_generated_texts.append(generated_text)   # every row's text, kept or rejected
            results.append(self.finish_example(text, generated_text, steps, ""))
        print(f"\tGenerated {len(texts)} examples together -> took {time() - start_time:.2f} seconds.")
        return results

    def generate_texts(self, prompts: List[str], max_new_tokens: int = 200, temperature: float = 0.0,
                       stop_strings: Optional[List[str]] = None, seed: int = 0,
                       prefixes: Optional[List[str]] = None) -> List[str]:
        """Plain text from the teacher for whole prompts (already formatted, no domain prompt
        added): no logits kept, no vocabulary check. For helper passes such as labelling gold
        solutions or writing pupil questions, and for running Gemma inside the tutor harness.
        temperature 0 = greedy; above 0 = sampled. A prefix starts the model's own turn (the
        harness writes "<solution>" once the route is decided); it is not part of the result."""
        texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                 if self.chat_template else prompt for prompt in prompts]
        if prefixes:
            texts = [text + prefix for text, prefix in zip(texts, prefixes)]
        inputs = self.encode_prompts(texts)
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
        if temperature > 0:
            torch.manual_seed(seed)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **sampling,
                stopping_criteria=self.stopping_criteria(stop_strings),
                eos_token_id=sorted(self.end_token_ids),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        results = []
        for row in output[:, inputs["input_ids"].shape[1]:]:
            text = self.tokenizer.decode(row, skip_special_tokens=False)
            for end in ("<turn|>", "<end_of_turn>", "<eos>", "<pad>"):
                text = text.split(end)[0]
            for scaffold in THOUGHT_SCAFFOLDS:
                if text.startswith(scaffold):
                    text = text[len(scaffold):]
            results.append(text.strip())
        return results

    def score_training_examples(self, texts: List[str], targets: List[str]) -> List[Tuple[Optional[Dict], Optional[str]]]:
        """Teacher-forced: the teacher reads a given target (a gold answer) after the prompt in
        one forward pass, and its distribution over the reduced vocabulary is kept at every
        target position. No generation, so the targets stay correct; the teacher supplies the
        soft labels. Rows are left-padded with explicit positions, and only the last positions'
        logits are computed (logits_to_keep), so the full vocabulary is never materialised for
        the prompt."""
        start_time = time()
        prompt_ids = [self.prompt_token_ids(self.build_prompt(text)) for text in texts]
        target_ids = [self.tokenizer(target, add_special_tokens=False)["input_ids"] for target in targets]
        sequences = [prompt + target for prompt, target in zip(prompt_ids, target_ids)]
        length = max(len(sequence) for sequence in sequences)
        keep = max(len(target) for target in target_ids) + 1
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        device = get_device()
        input_ids = torch.tensor([[pad] * (length - len(s)) + s for s in sequences], device=device)
        attention_mask = torch.tensor([[0] * (length - len(s)) + [1] * len(s) for s in sequences], device=device)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        token_list = self.vocabulary["token_list"]
        reduced_ids = torch.tensor([self.vocabulary["token_to_id"][token] for token in token_list], device=device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                logits_to_keep=keep).logits
            reduced = logits[:, :, reduced_ids].float()
            # the top token per position on the card (2026-11-09: the Python max over every row cost ~25% of a
            # scoring pass); the same index as before, ties to the first
            top_indices = reduced.argmax(dim=-1).cpu()
            reduced = reduced.cpu()

        results = []
        for row, (text, target, ids) in enumerate(zip(texts, targets, target_ids)):
            # the logits predicting target token j sit at window position keep - len(ids) - 1 + j
            offset = keep - len(ids) - 1
            steps, agree = [], 0
            for j, token_id in enumerate(ids):
                logit_vector = reduced[row, offset + j].tolist()
                predicted = int(top_indices[row, offset + j])
                token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                agree += token_list[predicted] == token
                steps.append({"token": token, "logits": logit_vector, "predicted_token_index": predicted})
            example, reason = self.finish_example(text, target, steps, "")
            if example is not None:
                example["teacher_agreement"] = round(agree / max(len(ids), 1), 4)
            results.append((example, reason))
        print(f"\tScored {len(texts)} gold targets together -> took {time() - start_time:.2f} seconds.")
        return results

    def finish_example(self, text: str, generated_text: str, steps: List[Dict], thinking: str) -> Tuple[Optional[Dict], Optional[str]]:
        """The checks every generated example passes before it is kept. They read the stored step
        tokens (what training learns from), not a re-tokenisation of the text: the teacher can
        generate a token split the tokenizer would not produce, whose pieces lie outside the
        reduced vocabulary while the re-tokenised text looks fine."""
        self.last_generated_text = generated_text
        unexpected_tokens = [step["token"] for step in steps if step["token"] not in self.response_tokens]
        if unexpected_tokens:
            print(f"\tSkipped: Generated response contains unexpected tokens: {unexpected_tokens}")
            return None, "unexpected_tokens"
        if len(steps) > self.max_generation_steps:
            print(f"\tSkipped: Generated response is too long ({len(steps)} tokens).")
            return None, "too_long"
        example = {"sentence": text, "model_response": generated_text, "steps": steps}
        if self.thinking:
            example["thinking"] = thinking
        return example, None

    def generate_single_step(self, pending: torch.Tensor, cache: DynamicCache,
                             total_length: int) -> Tuple[str, str, List[float], int, int]:
        """Feed the not-yet-cached tokens (`pending`) through the model with the KV cache and
        return the next token plus the full logit vector over the reduced vocabulary."""
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=pending.device)
        with torch.no_grad():
            output = self.model(input_ids=pending, attention_mask=attention_mask,
                                past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1, :]

        logit_vector = extract_logits_as_vector(logits, self.vocabulary)
        predicted_token_index = int(max(range(len(logit_vector)), key=lambda i: logit_vector[i]))

        next_token_id = torch.argmax(logits).item()
        token_repr = self.tokenizer.convert_ids_to_tokens([next_token_id])[0]
        token_decoded = self.tokenizer.decode([next_token_id])

        return token_repr, token_decoded, logit_vector, predicted_token_index, next_token_id

    def get_vocabulary(self) -> dict:
        return self.vocabulary

    def get_vocab_size(self) -> int:
        return self.vocabulary['vocab_size']

    def get_positions(self) -> dict:
        return self.vocabulary['positions']
