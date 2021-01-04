import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads, device):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads

        self.V = nn.Linear(self.emb_size, self.emb_size, bias = False)
        self.K = nn.Linear(self.emb_size, self.emb_size, bias = False)
        self.Q = nn.Linear(self.emb_size, self.emb_size, bias = False)
        self.fc_out = nn.Linear(self.emb_size, self.emb_size)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, value, key, query, mask):
        N = query.shape[0]

#        v_len, k_len, q_len = value.shape[1], key.shape[1], query.shape[1]

        value = self.V(value)
        key = self.K(key)
        query = self.Q(query)

        value = value.view(N, -1, self.heads, self.head_dim) #batch_size, v_len, n_heads, heads_dim
        key = key.view(N, -1, self.heads, self.head_dim) #batch_size, k_len, n_heads, heads_dim
        query = query.view(N, -1, self.heads, self.head_dim) #batch_size, q_len, n_heads, heads_dim

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])/self.scale #batch_size, n_heads, q_len, k_len

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention  = torch.softmax(energy, dim = -1)#normalized attention score for source sentence(key)
        # attention : batch_size, n_heads, q_len, k_len
        # value : batch_size, v_len, n_heads, head_dim

        out = torch.einsum("nhqa,nahd->nqhd",[attention, value]).contiguous()
        # out : (batch_size, q_len, n_heads, head_dim)
        out = out.view(N, -1, self.emb_size)
        #out : (batch_size, q_len, n_heads, head_dim) ->(batch_size, q_len, emb_size)

        out = self.fc_out(out)
        return out, attention

    @property
    def head_dim(self):
        assert (self.emb_size % self.heads ==0), "Divison Result tpye must be integer"
        return self.emb_size // self.heads

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(emb_size, heads, device)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        #v, q, k : batch_size, qkv_len, emb_size
        out, attention = self.attention(value, key, query, mask) #(batch_size, q_len, emb_size)
        x = self.dropout(self.norm1(out + query))
        x_forward = self.feed_forward(x)
        output = self.dropout(self.norm2(x_forward + x))
        return output, attention #batch_size, q_len, emb_size

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, emb_size, n_layers,
                 heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.input_Embedding = nn.Embedding(src_vocab_size, self.emb_size)
        self.positional_Embedding = nn.Embedding(max_len, self.emb_size)

        self.layers = nn.ModuleList([
            TransformerBlock(emb_size, heads, dropout, forward_expansion, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, src_len = x.shape
        positions = torch.arange(0, src_len).expand(batch_size, src_len).to(self.device)
        out = self.dropout(self.input_Embedding(x) + self.positional_Embedding(positions))

        for layer in self.layers:
            out, _ = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    """Add Masked Multi-Head Attention Layer
    at the bottom of TransformerBlock above"""
    def __init__(self, emb_size, heads, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(emb_size, heads, device)
        self.norm = nn.LayerNorm(emb_size)
        self.transformer_block = TransformerBlock(emb_size, heads, dropout, forward_expansion, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        out, _ = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(out + x))
        output, attention = self.transformer_block(value, key, query, src_mask)
        #query from Decoder Block
        #key and values from Encoder Block
        return output, attention #batch_size, q_len, emb_size

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, emb_size, n_layers,
                 heads, device, forward_expansion, dropout, max_len):
        super(Decoder, self).__init__()
        self.device = device
        self.output_Embedding = nn.Embedding(trg_vocab_size, emb_size)
        self.positional_Embedding = nn.Embedding(max_len, emb_size)
        self.layers = nn.ModuleList([
            DecoderBlock(emb_size, heads, dropout, forward_expansion, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        batch_size, trg_len = x.shape
        positions = torch.arange(0, trg_len).expand(batch_size, trg_len).to(self.device)
        x = self.dropout(self.output_Embedding(x) + self.positional_Embedding(positions))
        #x : (batch_size, trg_len, emb_size)
        for layer in self.layers:
            x, attention = layer(x, encoder_output, encoder_output, src_mask, trg_mask)
        out = self.fc_out(x)
        # output : (batch_size, trg_len, trg_vocab_size)
        # attention : (batch_size, n_heads, trg_len, src_len)
        return out, attention

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 emb_size = 256, n_layers = 6, forward_expansion = 4, heads = 8,
                 dropout = 0., device = "cuda:0", max_len = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, emb_size, n_layers, heads, device, forward_expansion, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, emb_size, n_layers, heads, device, forward_expansion, dropout, max_len)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask : (batch_size, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask : (batch_size, 1, 1, trg_len)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask : (trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_output = self.encoder(src, src_mask)
        #(batch_size, 1, 1, src_len) masking on (batch_size, n_heads, q_len, k_len) energy
        out, attention = self.decoder(trg, encoder_output, src_mask, trg_mask)
        #(batch_size, 1, trg_len, trg_len) masking on (batch_size, n_heads, q_len, k_len)
        return out, attention
