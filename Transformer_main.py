import torch
import torch.nn as nn
from utils import initialize_weights, epoch_time
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from train_eval import train, evaluate
import Transformer
import math
import time
import os
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import random
import spacy

# parser = argparse.ArgumentParser(description='Transformer Tutorial')
# parser.add_argument("--lr", default = 5e-4, type = float)
# parser.add_argument("--batch_size", default = 64, type= int)
# parser.add_argument("")
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')
batch_size = 64
lr = 5e-4
log_dir = './runs'
ckpt_dir = './ckpt'
n_epochs = 20
dropout = 0.1
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

de_field = Field(tokenize = tokenize_de, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True)
en_field = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True)

data_train, data_val, data_test = Multi30k.splits(exts = ('.de', '.en'), fields = (de_field, en_field))

de_field.build_vocab(data_train, min_freq = 2)
en_field.build_vocab(data_train, min_freq = 2)
print("Vocab building finished")

train_loader, val_loader, test_loader = BucketIterator.splits((data_train, data_val, data_test),batch_size = batch_size, device = device)

src_vocab_size = len(de_field.vocab)
trg_vocab_size = len(en_field.vocab)
src_pad_idx = de_field.vocab.stoi[de_field.pad_token]
trg_pad_idx = en_field.vocab.stoi[en_field.pad_token]

Seq2Seq_transformer = Transformer.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, dropout = dropout, n_layers = 3, forward_expansion = 2, device = device).to(device)
model_name = 'Transformer.pt'

print("Initialize weights")
Seq2Seq_transformer.apply(initialize_weights)

optimizer = torch.optim.Adam(Seq2Seq_transformer.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
writer = SummaryWriter(log_dir)
best_val_loss = float('inf')

for epoch in range(n_epochs):
    s = time.time()

    train_loss = train(Seq2Seq_transformer, train_loader, optimizer, criterion, clip=1)
    val_loss = evaluate(Seq2Seq_transformer, val_loader, criterion)

    t = time.time()

    epoch_min, epoch_sec = epoch_time(s, t)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(Seq2Seq_transformer.state_dict(), os.path.join(ckpt_dir, model_name))

    print("Epcoh : %02d | Elapsed Time : %02d min %02d sec" % (epoch + 1, epoch_min, epoch_sec))
    print("\t Train Loss : %.3f | Train PPL : %7.3f" % (train_loss, math.exp(train_loss)))
    print("\t Val   Loss : %.3f | Val PPL : %7.3f" % (val_loss, math.exp(val_loss)))
    writer.add_scalars(''.join([str(model_name), '/Train and Val Loss']),
                       {"Train Loss": train_loss, "Val Loss": val_loss}, epoch + 1)
    writer.add_scalars(''.join([str(model_name), '/Train and Val PPL']),
                       {'Train PPL': math.exp(train_loss), 'Val PPL': math.exp(val_loss)}, epoch + 1)
writer.close()



