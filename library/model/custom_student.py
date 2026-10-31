"""A custom student network trained by the same framework as the built-in Transformer.

    from library.model.custom.lstm_student import LstmStudent
    TrainingRun(domain=..., run_name="lstm-trial", model=LstmStudent, model_settings={"hidden": 384})

What the class must do (the whole contract):
  - __init__(input_vocab_size, output_vocab_size, **model_settings): the framework fills in the two sizes
    from the run's vocabularies.
  - forward(input_ids) takes [batch, seq] compact input ids and returns [batch, seq, output_vocab_size]
    logits: one guess per position over the reduced output words.
  - The guess at position t may only use tokens 0..t: training feeds the whole reply at once, so a model
    that looks ahead would cheat. Both rules are checked once, before training starts.
  - It is defined at the top level of an importable file (library/model/custom/ is the place), so a
    finished run can be rebuilt for the analysis and the demo.

Everything else is the framework's: the tokenizer and both vocabularies (StudentVocabularies), the losses,
schedules, checkpoints, stop / resume, logs and tests. Not available to a custom model: attention noise
(ignored, with a warning) and the key/value cache of the per-pass test (it decodes with SimpleDecoder,
which re-reads the reply at every step: slower, but right for any model).
"""

import torch
import torch.nn as nn

from library.model.student_vocabularies import StudentVocabularies
from library.specs import load_model_class

CHECK_LENGTH = 6   # tokens in the input the contract is checked on


class CustomStudent(StudentVocabularies, nn.Module):
    def __init__(self, model_class: type, model_settings: dict, domain, teacher_model: str,
                 auxiliary_token_percentage: float = 1.0, input_vocabulary: dict = None):
        super().__init__()
        self._set_up_vocabularies(domain, teacher_model, auxiliary_token_percentage, input_vocabulary)
        self.model_name = model_class.__name__
        self.model_settings = dict(model_settings or {})
        self.network = model_class(input_vocab_size=self.input_vocab_size, output_vocab_size=self.output_vocab_size,
                                   **self.model_settings)
        self.warned_about_attention_noise = False
        self._check_contract()
        print(f"Custom student {self.model_name}({self.model_settings}): {self.get_num_parameters():,} parameters, "
              f"input vocab {self.input_vocab_size:,}, output vocab {self.output_vocab_size:,}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.network(input_ids)

    def _check_contract(self) -> None:
        """The output shape, and that no guess looks ahead: changing the last input token must leave
        every earlier position's logits as they were."""
        input_ids = (torch.arange(CHECK_LENGTH) % self.input_vocab_size).unsqueeze(0)
        changed_ids = input_ids.clone()
        changed_ids[0, -1] = (changed_ids[0, -1] + 1) % self.input_vocab_size
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.network(input_ids)
            changed_logits = self.network(changed_ids)
        self.train(was_training)
        expected_shape = (1, CHECK_LENGTH, self.output_vocab_size)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(f"{self.model_name}.forward returned logits of shape {tuple(logits.shape)} for input ids "
                             f"of shape {tuple(input_ids.shape)}; expected {expected_shape} "
                             f"([batch, seq, output_vocab_size])")
        if not torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-5):
            raise ValueError(f"{self.model_name} looks ahead: changing the last input token changed the guesses at "
                             f"earlier positions, so training on whole replies would let it read the answer")

    def set_attention_noise(self, sigma: float) -> None:
        if sigma and not self.warned_about_attention_noise:
            print(f"Warning: attention noise is a built-in Transformer feature; {self.model_name} trains without it.")
            self.warned_about_attention_noise = True

    def load_weights(self, model_state_dict: dict) -> None:
        self.load_state_dict(model_state_dict, strict=True)

    def get_model_info(self) -> dict:
        return {
            "model": self.model_name,
            "total_parameters": self.get_num_parameters(),
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
        }


def build_student(settings: dict, domain, input_vocabulary: dict = None) -> nn.Module:
    """The run's student from its settings (a TrainingRun's as_dict, or a finished run's info.json):
    the custom model named in settings["model"], else the built-in Transformer."""
    if settings.get("model"):
        return CustomStudent(load_model_class(settings["model"]), settings.get("model_settings"), domain,
                             settings["teacher_model"], settings.get("auxiliary_token_percentage", 1.0),
                             input_vocabulary)
    from library.model.transformer import Transformer
    return Transformer(
        domain=domain,
        teacher_model=settings["teacher_model"],
        hidden_dim=settings["hidden_dim"],
        num_layers=settings["num_layers"],
        num_heads=settings["num_heads"],
        dropout=settings.get("dropout", 0.15),
        auxiliary_token_percentage=settings.get("auxiliary_token_percentage", 1.0),
        input_vocabulary=input_vocabulary,
    )
