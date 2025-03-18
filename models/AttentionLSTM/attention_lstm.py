import torch
import torch.nn as nn


class AttentionLSTMmodel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 output_size, 
                 output_length=1*24, 
                 batch_first=True, 
                 bidirectional=False,
                 device=None):
        super().__init__()
        self.output_length = output_length
        self.hidden_size = hidden_size
        
        # LSTM层
        # IN: (batch_size, seq_length, input_size)
        # OUT: (batch_size, seq_length, hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        
        # Attention 层
        # IN: (batch_size, seq_length, hidden_size)
        # OUT: (batch_size, output_length, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
        
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.activation = nn.ReLU()
        
    def attention(self, lstm_output):
        """
        计算注意力权重并应用到LSTM输出
        Args:
            lstm_output: shape (batch_size, seq_length, hidden_size)
        Returns:
            context_vector: shape (batch_size, 1, hidden_size)
        """
        # 计算Q、K、V
        # (batch_size, output_length, hidden_size)
        Q = self.activation(self.query(lstm_output[:, -self.output_length:, :]))
        # (batch_size, seq_length, hidden_size)
        K = self.activation(self.key(lstm_output))
        # (batch_size, seq_length, hidden_size)
        V = self.activation(self.value(lstm_output))
        
        # 计算注意力分数
        # (batch_size, output_length, hidden_size) × (batch_size, hidden_size, seq_length)
        # -> (batch_size, output_length, seq_length)
        attention_weights = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(attention_weights, dim=2)
        
        # 将注意力权重应用到V
        # (batch_size, output_length, seq_length) × (batch_size, seq_length, hidden_size)
        # -> (batch_size, output_length, hidden_size)
        context_vector = torch.bmm(attention_weights, V)
        
        return context_vector

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量, shape (batch_size, seq_length, input_size)
        Returns:
            output: shape (batch_size, output_length, output_size)
        """
        # LSTM 层
        # (batch_size, seq_length, input_size) -> (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Attention 层
        # (batch_size, seq_length, hidden_size) -> (batch_size, output_length, hidden_size)
        context_vector = self.attention(lstm_out)
        
        # MLP 层 + residual connection
        mlp_input = context_vector + lstm_out[:, -self.output_length:, :]
        output = self.mlp(mlp_input)

        return output