import torch.nn as nn
import torch


class AttentionLSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_length=1*24, batch_first=True, bidirectional=False):
        super().__init__()
        self.output_length=output_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -self.output_length:, :]
        e = torch.matmul(self.attention(out), out.permute(0, 2, 1))

        attention = torch.softmax(e, dim=-1)
        out = torch.bmm(attention, out)
        out = self.relu(out)
        out = self.fc(out)
        return out
