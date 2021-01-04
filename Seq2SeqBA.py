import torch
import torch.nn as nn
import torch.nn.functional as F
import random
class Encoder(nn.Module):
    def __init__(self, i_dim, e_dim, enc_h_dim, dec_h_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(i_dim, e_dim)
        self.rnn = nn.GRU(e_dim, enc_h_dim, bidirectional=True)
        self.fc = nn.Linear(enc_h_dim * 2, dec_h_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x)) #(src_len, batch_size, e_dim)
        outputs, hidden = self.rnn(embedded)
        #outputs : (src_len, batch_size, enc_h_dim * 2)
        #hidden : (n_layers * 2, batch_size, enc_h_dim)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #hidden : (batch_size, enc_h_dim * 2)->(batch_size, dec_h_dim)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_h_dim, dec_h_dim):
        super().__init__()
        self.attn = nn.Linear((enc_h_dim * 2) + dec_h_dim, dec_h_dim)
        self.v = nn.Linear(dec_h_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        #hidden : (batch_size, dec_h_dim)
        #encoder_outputs : (src_len, batch_size, enc_h_dim * 2)

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) #(batch_size, src_len, dec_h_dim)
        encoder_outputs = encoder_outputs.permute(1, 0, 2) #(batch_size, src_len, enc_h_dim * 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        #energy : (batch_size, src_len, dec_h_dim)

        attention = self.v(energy).squeeze(2) #(batch_size, src_len)

        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, o_dim, e_dim, enc_h_dim, dec_h_dim, dropout, attention):
        super().__init__()
        self.o_dim = o_dim
        self.attention = attention
        self.embedding = nn.Embedding(o_dim, e_dim)
        self.rnn = nn.GRU((enc_h_dim * 2) + e_dim, dec_h_dim)
        self.fc = nn.Linear((enc_h_dim * 2) + dec_h_dim + e_dim, o_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        #input : (batch_size,)
        #encoder_outputs : (src_len, batch_size, enc_h_dim * 2)
        #hidden : (batch_size, dec_h_dim)

        input = input.unsqueeze(0) #(1, batch_size)
        embedded = self.dropout(self.embedding(input)) #(1, batch_size, e_dim)

        attention_score = self.attention(hidden, encoder_outputs)
        #attention_score : (batch_size, src_len)

        attention_score = attention_score.unsqueeze(1) #(batch_size, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2) #(batch_size, src_len, enc_h_dim * 2)

        weighted = torch.bmm(attention_score, encoder_outputs)#(batch_size, 1, enc_h_dim * 2)
        weighted = weighted.permute(1, 0, 2) #(1, batch_size, enc_h_dim * 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input : (1, batch_size, enc_h_dim * 2 + e_dim)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output : (1, batch_size, dec_h_dim)
        #hidden : (1, batch_size, dec_h_dim)

        assert (output == hidden).all()

        embedded = embedded.squeeze(0) #(batch_size, e_dim)
        output = output.squeeze(0) #(batch_size, dec_h_dim)
        weighted = weighted.squeeze(0) #(batch_size, enc_h_dim * 2)

        prediction = self.fc(torch.cat((embedded, output, weighted), dim = 1))
        #prediction : (batch_size, o_dim)

        return prediction, hidden.squeeze(0)

class BahdanauS2S(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio = 0.5):

        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = self.decoder.o_dim

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(source)

        input = target[0,:]

        for t in range(1, target_length):
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(dim = 1)

            input = target[t] if teacher_force else top1

        return outputs
