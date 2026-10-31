"""An example custom student: a plain LSTM (causal by nature: it reads left to right).

    TrainingRun(domain=..., run_name="lstm-trial", model=LstmStudent,
                model_settings={"hidden": 384, "layers": 2, "dropout": 0.1})
"""

from torch import nn


class LstmStudent(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
        self.output = nn.Linear(hidden, output_vocab_size)

    def forward(self, input_ids):
        hidden_states, _ = self.lstm(self.embedding(input_ids))
        return self.output(hidden_states)
