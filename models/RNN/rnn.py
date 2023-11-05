import torch.nn as nn

class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_length=1 * 24, batch_first=True, bidirectional=False):
        super().__init__()
        self.output_length = output_length
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out[:, -self.output_length:, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out