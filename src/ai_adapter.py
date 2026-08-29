import torch
import torch.nn as nn


class AIAdapter(nn.Module):

    def __init__(self, input_size=6, hidden_size=32):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            3
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        correction = self.fc(last_output)

        return correction