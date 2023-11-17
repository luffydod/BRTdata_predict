import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, (hidden_state, cell_state) = self.lstm(x, (h0, c0))
        return out, (hidden_state, cell_state)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        out, (hidden, cell) = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out)
        return out, (hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_sequence, target_sequence, teacher_forcing_ratio=1):
        encoder_output, (hidden_state, cell_state) = self.encoder(input_sequence)
        # 初始化解码器的输入
        decoder_input = torch.zeros(input_sequence.size(0), 1, target_sequence.size(2)).to(input_sequence.device)
        # decoder_input = torch.randn(input_sequence.size(0), 1, target_sequence.size(2)).to(input_sequence.device)
        # decoder_input = target_sequence[:, 0:1, :]

        # 存储预测结果
        predicted_outputs = []

        # 是否使用教师强制
        use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

        # 初始化解码器的隐藏状态和细胞状态
        decoder_hidden = hidden_state
        decoder_cell = cell_state
        
        # 进行解码器的前向传播

        for t in range(target_sequence.size(1)):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            predicted_outputs.append(decoder_output)

            # 如果使用教师强制，则下一个时间步的解码器输入为真实目标序列；否则使用模型生成的输出
            decoder_input = target_sequence[:, t:t+1, :] if use_teacher_forcing else decoder_output

        # 将预测结果转换为三维张量
        predicted_outputs = torch.cat(predicted_outputs, dim=1)

        return predicted_outputs