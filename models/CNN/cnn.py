import torch.nn as nn

class CNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_length=1*24) -> None:
        super().__init__()
        self.output_length = output_length
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 输入数据形状为(samples, sequence_length, features)
        x = x.permute(0, 2, 1)  # (samples, sequence_length, features) -> (samples, features, sequence_length)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (samples, features, sequence_length) -> (samples, sequence_length, features)
        x = self.fc(x[:, -self.output_length:, :])
        return x