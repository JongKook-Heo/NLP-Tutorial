import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, i_dim, e_dim, h_dim, n_layers, dropout):
        super().__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(i_dim, e_dim)
        self.rnn = nn.LSTM(e_dim, h_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout()

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs :(src_len, batch_size, h_dim * n_directions)
        #hidden, cell : (n_layers * n_directions, batch_size, h_dim)

        return hidden, cell #context vector


class Decoder(nn.Module):
    def __init__(self, o_dim, e_dim, h_dim, n_layers, dropout):
        super().__init__()
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(o_dim, e_dim)
        self.rnn = nn.LSTM(e_dim, h_dim, n_layers, dropout = dropout)

        self.fc = nn.Linear(h_dim, o_dim)
        self.dropout = nn.Dropout()

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0) #(1, batch)
        embedded = self.dropout(self.embedding(input)) #(1, batch, e_dim)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output : (1, batch_size, h_dim)
        #hidden, cell : (n_layers * n_directions, batch_size, h_dim)

        prediction = self.fc(output.squeeze(0))
        #prediction : (batch_size, o_dim)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.h_dim == decoder.h_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, source, target, teacher_forcing_ratio = 0.5):

        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.o_dim

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        input = target[0, :]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(dim = 1)
            input = target[t] if teacher_force else top1

        return outputs