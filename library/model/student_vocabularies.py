"""The framework's side of every student, whatever its network: the teacher's tokenizer, the
compact input words (with the unknown-token row) and the reduced output words. The built-in
Transformer and a custom model's CustomStudent both take it on, so the trainer, the evaluator and
StudentModel see the same attributes and methods on either. It holds no weights, so a checkpoint's
contents do not depend on it."""

from transformers import AutoTokenizer

from library.domain import as_domain
from library.shared.vocabulary import build_vocabulary


class StudentVocabularies:
    def _set_up_vocabularies(self, domain, teacher_model: str, auxiliary_token_percentage: float = 1.0,
                             input_vocabulary: dict = None) -> None:
        self.domain = as_domain(domain)

        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model)
        self.full_input_vocab_size = self.tokenizer.vocab_size

        if input_vocabulary is not None:
            self.input_vocab_size = input_vocabulary['compact_vocab_size']
            self.input_token_mapping = {
                int(gemma_id): compact_id
                for gemma_id, compact_id in input_vocabulary['gemma_id_to_compact_id'].items()
            }
            # The row every token never seen in the batches maps to (vocabularies built
            # before it existed have none, and an unseen token then raises).
            self.unknown_input_id = input_vocabulary.get('unknown_compact_id')
        else:
            self.input_vocab_size = self.full_input_vocab_size
            self.input_token_mapping = None
            self.unknown_input_id = None

        self.vocabulary = build_vocabulary(
            self.tokenizer, self.domain, auxiliary_token_percentage)
        self.output_vocab_size = self.vocabulary['vocab_size']

        self.output_token_ids = [
            self.vocabulary['token_to_id'][token]
            for token in self.vocabulary['token_list']
        ]

    def remap_input_tokens(self, token_ids: list) -> list:
        if self.input_token_mapping is None:
            return token_ids
        remapped = []
        for token_id in token_ids:
            if token_id in self.input_token_mapping:
                remapped.append(self.input_token_mapping[token_id])
            elif self.unknown_input_id is not None:
                remapped.append(self.unknown_input_id)
            else:
                raise ValueError(
                    f"Token ID {token_id} not found in input vocabulary mapping, and this vocabulary "
                    f"has no unknown-token row. Rebuild it with: python -m library.shared.build_input_vocabulary"
                )
        return remapped

    def get_vocabulary(self) -> dict:
        return self.vocabulary

    def get_positions(self) -> dict:
        return self.vocabulary['positions']

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
