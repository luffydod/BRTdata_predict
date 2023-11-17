import torch.nn as nn

class LSTMmodel(nn.Module):
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
        if(bidirectional==True):    # 双向LSTM，隐状态翻倍
            self.fc = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -self.output_length:, :])
        return out