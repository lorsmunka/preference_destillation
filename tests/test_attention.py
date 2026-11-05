"""library/tooling/attention.py: the student's attention maps are the ones its forward pass really uses
(the logits rebuilt from them equal the model's own), and the teacher hooks catch every layer of a
Gemma 4 text model (a tiny one with random weights: the real teacher is never loaded here).
CPU only. Needs the Gemma tokenizer in the local Hugging Face cache (no network).

    python -m unittest discover -s tests -p "test_attention.py"
"""

import json
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from library.tooling.attention import TeacherAttention, group_columns, student_attention
from tests.test_training import TOKENIZER, SENTENCES, tiny_input_vocabulary
from library.domain import REDDIT_SENTIMENT
from library.model.transformer import Transformer


def small_student() -> Transformer:
    torch.manual_seed(0)
    model = Transformer(domain=REDDIT_SENTIMENT, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.1,
                        input_vocabulary=tiny_input_vocabulary())
    return model.eval()


def tiny_gemma4():
    """A Gemma 4 text model with the real one's layer pattern (sliding layers, then a full one with its
    wider head) and attention options, shrunk to a few thousand random weights."""
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM
    torch.manual_seed(0)
    config = Gemma4UnifiedTextConfig(
        vocab_size=300, vocab_size_per_layer_input=300, hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=5, max_position_embeddings=256,
        hidden_size_per_layer_input=0, final_logit_softcapping=30.0,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                            "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
    config._attn_implementation = "eager"
    return Gemma4UnifiedForCausalLM(config).eval()


class StudentAttentionTest(unittest.TestCase):
    def setUp(self):
        self.model = small_student()
        token_ids = TOKENIZER.encode(SENTENCES[0] + "\n\n", add_special_tokens=False)
        self.input_ids = torch.tensor([self.model.remap_input_tokens(token_ids)])

    def test_logits_rebuilt_from_the_maps_equal_the_forward_pass(self):
        logits, maps = student_attention(self.model, self.input_ids)
        with torch.no_grad():
            expected = self.model(self.input_ids)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))
        seq_length = self.input_ids.shape[1]
        self.assertEqual(tuple(maps.shape), (2, 2, seq_length, seq_length))

    def test_rows_add_up_to_one_and_never_look_ahead(self):
        _, maps = student_attention(self.model, self.input_ids)
        self.assertTrue(torch.allclose(maps.sum(-1), torch.ones(maps.shape[:-1]), atol=1e-5))
        self.assertEqual(float(torch.triu(maps, diagonal=1).abs().sum()), 0.0)

    def test_chosen_rows_and_grouped_columns(self):
        _, maps = student_attention(self.model, self.input_ids)
        seq_length = self.input_ids.shape[1]
        queries = [seq_length - 3, seq_length - 1]
        groups = [0] * 2 + [1] * (seq_length - 2)   # the first two positions together, the rest together
        _, grouped = student_attention(self.model, self.input_ids, queries, groups, 2)
        self.assertTrue(torch.allclose(grouped[..., 0], maps[:, :, queries, :2].sum(-1), atol=1e-6))
        self.assertTrue(torch.allclose(grouped.sum(-1), torch.ones(grouped.shape[:-1]), atol=1e-5))
        self.assertTrue(torch.allclose(group_columns(maps, list(range(seq_length)), seq_length), maps))


class TeacherAttentionTest(unittest.TestCase):
    def test_hooks_catch_every_layer_like_output_attentions(self):
        model = tiny_gemma4()
        input_ids = list(range(2, 14))
        queries = [3, 8, 11]
        rows = TeacherAttention(model).rows(input_ids, queries)
        self.assertEqual(tuple(rows.shape), (6, 4, 3, 12))
        with torch.no_grad():
            reference = model(input_ids=torch.tensor([input_ids]), output_attentions=True).attentions
        for layer, layer_rows in enumerate(rows):
            self.assertTrue(torch.allclose(layer_rows, reference[layer][0][:, queries].float(), atol=1e-6))
        self.assertTrue(torch.allclose(rows.sum(-1), torch.ones(rows.shape[:-1]), atol=1e-4))
        # sliding layers (window 5) put nothing further back than the window; the full layer may
        self.assertEqual(float(rows[0, :, 2, :11 - 4].sum()), 0.0)
        self.assertGreater(float(rows[5, :, 2, :11 - 4].sum()), 0.0)

    def test_without_eager_weights_it_fails_loudly(self):
        model = tiny_gemma4()
        model.config._attn_implementation = "sdpa"
        with self.assertRaises(RuntimeError):
            TeacherAttention(model).rows(list(range(2, 10)), [4])


if __name__ == "__main__":
    unittest.main()
