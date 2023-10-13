# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import numpy as np

BOS_token = 1
EOS_token = 2


# step 2: define encoder
class Encoder(nn.Module):
    def __init__(self, V, E, H):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(V, E, max_norm=1, scale_grad_by_freq=True)
        self.gru = nn.GRU(E, H, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, T = x.shape
        mask_lens = (x != 0).sum(1).to(torch.device("cpu"))
        emb = self.embedding(x)
        x_pack = pack_padded_sequence(
            emb, mask_lens, batch_first=True, enforce_sorted=False
        )
        all_h_packed, _ = self.gru(x_pack)
        all_h, _ = pad_packed_sequence(all_h_packed, batch_first=True)
        return all_h
        # return all hidden states

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)


# enc = Encoder(8,4,2)
# x = [[4,2,2],[3,5]]
# out = enc(x)
# %%
# create AttentionGRU


class AddAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(2 * hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys):
        """
        keys dim: B,Tx,H
        query dim: B,1,H
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(-1)
        scores = scores.unsqueeze(1)
        # fix the weights
        enc_mask = torch.all((keys == 0), dim=-1).unsqueeze(1)
        scores.masked_fill(enc_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)  # B,1,Tx
        context = weights @ keys  # B,1,H

        return context, weights


# attn = AddAttention(4)
# q = torch.randn(4,1,4)
# k = torch.randn(4,2,8)
# out = attn(q,k)
# %%


class Decoder(nn.Module):
    def __init__(self, V, E, H):
        super().__init__()
        self.embedding = nn.Embedding(V, E)
        self.attention = AddAttention(H)
        self.gru = nn.GRU(E + 2 * H, H, batch_first=True)
        self.out = nn.Linear(H, V)
        self.initialW = nn.Parameter(torch.randn(H, H))

        # buffers
        self.register_buffer("hidden_decoder", torch.zeros(1, 1, H))

    def forward(self, x, hidden_encoder):
        # made a change to be stashed
        H = self.gru.hidden_size
        x_emb = self.embedding(x)
        B, T, E = x_emb.shape
        s_prev = hidden_encoder[:, :1, H:] @ self.initialW  # shape B,1,H
        hidden_decoder = self.hidden_decoder.repeat(B, T, 1)
        for t in range(T):
            x_t = x_emb[:, t, :].unsqueeze(1)
            c_t, _ = self.attention(s_prev, hidden_encoder)
            x_in = torch.cat((x_t, c_t), dim=-1)
            s_prev, _ = self.gru(x_in)
            hidden_decoder[:, t, :] = s_prev.squeeze(1)

        out = self.out(hidden_decoder)

        return out

    def evaluate(self, x, hidden_encoder, s_prev):
        with torch.no_grad():
            x_emb = self.embedding(x)
            c_t, weights = self.attention(s_prev, hidden_encoder)
            x_in = torch.cat((x_emb, c_t), dim=-1)
            s_prev, _ = self.gru(x_in)
            out = self.out(s_prev)

        return out, s_prev, weights


# x = [[1,2,3],[2,5]]
# all_h = torch.randn(2,4,8)
# dec = Decoder(5,10,4)
# out = dec(x,all_h)
# %%


class TranslationNN(nn.Module):
    def __init__(self, V_s, V_t, E, H):
        super(TranslationNN, self).__init__()
        self.encoder = Encoder(V_s, E, H)
        self.decoder = Decoder(V_t, E, H)

        # register buffers

        self.register_buffer("x_init", torch.ones(1, 1).long())

    def forward(self, x_s, x_t):
        all_h_enc = self.encoder(x_s)
        out = self.decoder(x_t, all_h_enc)
        return out

    def evaluate(self, x_s, MAXLEN=30):
        with torch.no_grad():
            hidden_encoder = self.encoder.evaluate(x_s)
            B, T, H = hidden_encoder.shape
            H = H // 2  # using bidir
            x_t = self.x_init.repeat(B, 1)
            s_prev = hidden_encoder[:, :1, H:] @ self.decoder.initialW
            counter = 0
            outs = torch.zeros(B, MAXLEN)
            weights = torch.zeros(B, MAXLEN, T)
            while (
                not torch.all(torch.any(outs == EOS_token, dim=1), dim=0).item()
            ) and (counter < MAXLEN):
                out, s_prev, weight = self.decoder.evaluate(x_t, hidden_encoder, s_prev)
                probs = F.softmax(out, dim=-1)
                x_t = torch.argmax(probs, axis=-1)
                outs[:, counter] = x_t.squeeze(1)
                weights[:, counter, :] = weight.squeeze(1)
                counter += 1

        return outs, weights
