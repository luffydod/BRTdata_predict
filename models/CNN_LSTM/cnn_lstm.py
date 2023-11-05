import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, output_size, output_length=1*24, batch_first=True) -> None:
        super().__init__()
        self.output_length = output_length
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=batch_first)
        self.fc = nn.Linear(in_features=128, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据形状为(samples, sequence_length, features)
        x = x.permute(0, 2, 1)  # (samples, sequence_length, features) -> (samples, features, sequence_length)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (samples, features, sequence_length) -> (samples, sequence_length, features)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -self.output_length:, :])
        return x


