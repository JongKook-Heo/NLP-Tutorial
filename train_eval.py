import torch
import Transformer
import Seq2Seq
import Seq2SeqBA

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    if isinstance(model, (Seq2SeqBA.BahdanauS2S, Seq2Seq.Seq2Seq)):
        for idx, batch in enumerate(iterator):
            src = batch.src #(src_len, batch_size)
            trg = batch.trg #(trg_len, batch_size)

            optimizer.zero_grad()

            output = model(src, trg) #(trg_len, batch_size, o_dim)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss/len(iterator)
    elif isinstance(model, Transformer.Transformer):
        for idx, batch in enumerate(iterator):
            src = batch.src  # (batch_size, src_len)
            trg = batch.trg  # (batch_size, trg_len)

            optimizer.zero_grad()

            output, attention = model(src, trg[:, :-1])  # until before <eos> token
            # output : (batch_size, trg_len - 1, trg_vocab_size)
            # attention : (batch_size, n_heads, trg_len -1, src_len

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(iterator)
    else:
        raise NotImplementedError

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    if isinstance(model, (Seq2SeqBA.BahdanauS2S, Seq2Seq.Seq2Seq)):
        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()
        return epoch_loss / len(iterator)
    elif isinstance(model, Transformer.Transformer):
        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output, attention = model(src, trg[:,:-1])

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()
        return epoch_loss / len(iterator)
    else:
        raise NotImplementedError







