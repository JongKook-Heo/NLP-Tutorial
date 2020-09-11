import torch
import torch.nn as nn
from utils import initialize_weights, epoch_time
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from train_eval import train, evaluate
import Seq2Seq
import Seq2SeqBA
import math
import time
import os
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import random
import spacy

parser = argparse.ArgumentParser(description="NLP Tutorial By Seq 2 Seq")
parser.add_argument("--lr", default = 5e-3, type=float)
parser.add_argument("--batch_size", default = 64, type = int)
parser.add_argument("--num_epochs", default = 20, type = int)

parser.add_argument("--log_dir", default = "./runs")
parser.add_argument("--ckpt_dir", default = "./ckpt")

parser.add_argument("--att", default = False, type = bool)

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
att = args.att
log_dir = args.log_dir
ckpt_dir = args.ckpt_dir
device = torch.device("cuda")
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de_reverse(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if att == True:
    de_field = Field(tokenize= tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
else:
    de_field = Field(tokenize = tokenize_de_reverse, init_token='<sos>', eos_token='<eos>', lower = True)
en_field = Field(tokenize= tokenize_en, init_token='<sos>',eos_token= '<eos>', lower = True)

data_train, data_val, data_test = Multi30k.splits(exts=('.de', '.en'), fields= (de_field, en_field))


de_field.build_vocab(data_train, min_freq = 2)
en_field.build_vocab(data_train, min_freq = 2)
print("Vocab building finished")

train_loader, val_loader, test_loader = BucketIterator.splits((data_train, data_val, data_test), batch_size=batch_size, device = device)

i_dim = len(de_field.vocab)
o_dim = len(en_field.vocab)
enc_h_dim = 512
dec_h_dim = 512
e_dim = 256
dropout = 0.5

if att==True:
    attention = Seq2SeqBA.Attention(enc_h_dim, dec_h_dim)
    encoder = Seq2SeqBA.Encoder(i_dim, e_dim, enc_h_dim, dec_h_dim, dropout)
    decoder = Seq2SeqBA.Decoder(o_dim, e_dim, enc_h_dim, dec_h_dim, dropout, attention)
    model = Seq2SeqBA.BahdanauS2S(encoder, decoder, device).to(device)
    model_name = "S2SBA.pt"
else:
    encoder = Seq2Seq.Encoder(i_dim, e_dim, enc_h_dim, n_layers = 2, dropout = dropout)
    decoder = Seq2Seq.Decoder(o_dim, e_dim, dec_h_dim, n_layers = 2, dropout = dropout)
    model = Seq2Seq.Seq2Seq(encoder, decoder, device).to(device)
    model_name = "S2S.pt"

print("Initialize weights")
model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(), lr = lr)
target_pad_idx = en_field.vocab.stoi[en_field.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)

best_val_loss = float('inf')
writer = SummaryWriter(log_dir)
for epoch in range(num_epochs):
    s = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, clip= 1)
    val_loss = evaluate(model, val_loader, criterion)

    t = time.time()

    epoch_min, epoch_sec = epoch_time(s, t)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(ckpt_dir,model_name))

    print("Epoch : %02d | Elapsed Time : %02d min %02d sec"%(epoch + 1, epoch_min, epoch_sec))
    print("\t Train Loss : %.3f | Train PPL : %7.3f"%(train_loss, math.exp(train_loss)))
    print("\t Val   Loss : %.3f | Val   PPL : %7.3f"%(val_loss, math.exp(val_loss)))
    writer.add_scalars(''.join([str(model_name),'/Train and Val Loss']),{"Train_Loss" : train_loss, "Val_Loss" : val_loss}, epoch + 1)
    writer.add_scalars(''.join([str(model_name),'/Train and Val PPL']), {"Train PPL" : math.exp(train_loss), "Val PPL" : math.exp(val_loss)}, epoch + 1)

writer.close()







