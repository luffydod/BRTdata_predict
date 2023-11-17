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
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * output_length)
        self.relu = nn.ReLU()

    def attention(self, lstm_output):
        # 计算注意力权重
            # attn_weights: B, len, 1
        attn_weights = torch.softmax(self.fc1(lstm_output), dim=1)
        # 对LSTM输出进行加权平均
            # (B, 1, len) * (B, len, hidden_size)
        attn_applied = torch.bmm(attn_weights.permute(0, 2, 1), lstm_output)
        return attn_applied

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # 计算注意力权重并应用到LSTM输出
        attn_applied = self.attention(lstm_out)

        out = self.fc2(attn_applied)

        out = out.permute(0, 2, 1)
        return out
