import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, output_size, output_length=1*24, batch_first=True) -> None:
        super().__init__()
        self.output_length = output_length
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=batch_first)
        self.fc = nn.Linear(in_features=64, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据形状为(samples, sequence_length, features) [B, S, F]
        x = x.permute(0, 2, 1)  # [B, S, F] -> [B, F, S]
        x = self.conv1(x)       # [B, F, S] -> [B, out_channels_1, S]
        x = self.relu(x)
        x = self.maxpool(x)     # [B, out_channels_1, S] -> [B, out_channels_1, S-2]

        x = self.conv2(x)       # [B, out_channels_1, S-2] -> [B, out_channels_2, S-2]
        x = self.relu(x)
        x = self.maxpool(x)     # [B, out_channels_2, S-2] -> [B, out_channels, S-4]

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -self.output_length:, :])
        return x


