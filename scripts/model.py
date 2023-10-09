# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import numpy as np


# step 2: define encoder
class Encoder(nn.Module):
    def __init__(self,V,E,H):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(V,E,max_norm=1, scale_grad_by_freq=True)
        self.gru = nn.GRU(E,H,batch_first=True,bidirectional=True)

    def forward(self, x):
        x = pad_sequence((torch.tensor(i) for i in x),batch_first=True)
        B,T = x.shape
        mask_lens = (x!=0).sum(1).to(torch.device('cpu'))
        emb = self.embedding(x)
        x_pack = pack_padded_sequence(emb,mask_lens,batch_first=True,enforce_sorted=False)
        all_h_packed,_ = self.gru(x_pack)
        all_h, _ = pad_packed_sequence(all_h_packed,batch_first=True)
        return all_h
        # return all hidden states

    def evaluate(self,x):
        with torch.no_grad():
            return self.forward(x)
        
enc = Encoder(8,4,2)
x = [[4,2,2],[3,5]]
out = enc(x)
# %%
# create AttentionGRU

class AddAttention(nn.Module):

    def __init__(self,hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size,hidden_size)
        self.Ua = nn.Linear(2*hidden_size,hidden_size)
        self.Va = nn.Linear(hidden_size,1,bias=False)


    def forward(self,query,keys):
        """
        keys dim: B,Tx,H
        query dim: B,1,H
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(-1)
        scores = scores.unsqueeze(1)

        weights = F.softmax(scores,dim=-1) # B,1,Tx
        context = weights @ keys # B,1,H

        return context, weights 

attn = AddAttention(4)
q = torch.randn(4,1,4)
k = torch.randn(4,2,8)
out = attn(q,k)
#%%

class Decoder(nn.Module):
    def __init__(self,E,V,H):
        super().__init__()
        self.embedding = nn.Embedding(V,E)
        self.attention = AddAttention(H)
        self.gru = nn.GRU(E+2*H,H,batch_first=True)
        self.out = nn.Linear(H,V)
        self.initialW = nn.Parameter(torch.randn(H,H))

    def forward(self,x,hidden_encoder):
        x = pad_sequence((torch.tensor(i) for i in x),batch_first=True)
        H = self.gru.hidden_size
        x_emb = self.embedding(x)
        B,T,E = x_emb.shape
        s_prev = hidden_encoder[:,:1,H:] @ self.initialW # shape B,1,H
        hidden_decoder = torch.zeros(B,T,H)
        for t in range(T):
            x_t = x_emb[:,t,:].unsqueeze(1)
            c_t,_ = self.attention(s_prev,hidden_encoder)
            x_in = torch.cat((x_t,c_t),dim=-1)
            s_prev, _ = self.gru(x_in)
            hidden_decoder[:,t,:] = s_prev.squeeze(1)

        out = self.out(hidden_decoder)

        return out

x = [[1,2,3],[2,5]]
all_h = torch.randn(2,4,8)
dec = Decoder(5,10,4)
out = dec(x,all_h)
#%%
# decoder time

# class Decoder(nn.Module):
#     def __init__(self, V_t, E, H):
#         super(Decoder, self).__init__()
#         self.embedding = nn.Embedding(V_t, E, max_norm=1, scale_grad_by_freq=True)
#         self.attention = AttentionGRU(E,H)
#         self.fc = nn.Linear(H,V_t)

#         init.normal_(self.fc.weight,mean=0,std=0.1)
        
#     def forward(self, x, all_hidden_enc):
#         emb = self.embedding(x) # B,T,V 
#         all_hidden, _ = self.attention(emb,all_hidden_enc) # B,T,H
#         out = self.fc(all_hidden) # B,T,V
#         return out
    
#     def evaluate(self,x,all_hidden_enc,s_prev=None):
#         with torch.no_grad():
#             emb = self.embedding(x) # B,T,V but T will be 1
#             all_h_dec, st = self.attention.evaluate(emb,all_hidden_enc,s_prev)
#             out = self.fc(all_h_dec) # B,T,V but again T will be 1
#             return out,st

