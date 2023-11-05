import torch
import torch.nn as nn
import numpy as np

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_length=1*24, batch_first=True):
        super().__init__()
        self.output_length=output_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出并传递给全连接层
        out = self.fc(out[:, -self.output_length:, :])
        # out = out[:, -self.output_length:, :]
        # samples, out_length, hidden_size = out.shape
        # out = out.contiguous().view(-1, hidden_size)
        # out = self.fc(out)
        # out = out.view(samples, out_length, -1)
        return out