#! python3
#%%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from torchtext.data.metrics import bleu_score
from sacremoses import MosesTokenizer,MosesDetokenizer
import pandas as pd
from collections import Counter, namedtuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scripts.dataset import TranslationDataset # The logic of TranslationDataset is defined in the file dataset.py

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

token_en = MosesTokenizer(lang='en')
token_fr = MosesTokenizer(lang='fr')

df = pd.read_csv('data/fra.txt',sep='\t',header=None,names=['en','fr','attribution'])
df.drop('attribution',axis=1,inplace=True)

V_s = vocab_source = 2001
V_t = vocab_target = 5001
Dataset = TranslationDataset(df,V_s,V_t,from_file=True)

#%%
E = 128
H = 256
B = 256

# %%

# step 2: define encoder
class Encoder(nn.Module):
    def __init__(self,V,E,H):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(V,E,max_norm=1, scale_grad_by_freq=True)
        self.gru = nn.GRU(E,H,batch_first=True,bidirectional=True)

    def forward(self, x):
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
# %%
# create AttentionGRU
class AttentionGRU(nn.Module):
    
    def __init__(self,E,H):
        super(AttentionGRU, self).__init__()

        # weights for embedding
        self.input_r = nn.Linear(E,H)
        self.input_z = nn.Linear(E,H)
        self.input = nn.Linear(E,H)
        
        # weights for decoder hidden states
        self.hidden_r = nn.Linear(H,H)
        self.hidden_z = nn.Linear(H,H)
        self.hidden = nn.Linear(H,H)

        # for context vector (c)
        self.context_r = nn.Linear(2*H,H)
        self.context_z = nn.Linear(2*H,H)
        self.context = nn.Linear(2*H,H)

        # weights to calculate alpha
        self.dec_projection = nn.Linear(H,H)
        self.enc_projection = nn.Linear(2*H,H)
        self.energy_projection = nn.Linear(H,1,bias=False)

        # weight initialization
        self.init_weights()

    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param,nonlinearity='tanh',a=0.577) # dividing by root 3

    def forward(self, emb, all_h_enc,s_prev=None):
        # all_h_enc dimension will be B,Tx,H so if I need the mask
        enc_mask = torch.all((all_h_enc !=0),dim=-1,keepdim=True)
        B,T,E = emb.shape
        H = self.hidden.in_features
        s_prev = torch.zeros(B,H).to(device) if s_prev is None else s_prev
        encoder_energies = self.enc_projection(all_h_enc)
        all_hidden = torch.zeros(B,T,H).to(device)
        for t in range(T):
            # calculate the energies
            et = self.energy_projection(self.dec_projection(s_prev).unsqueeze(1) + encoder_energies)
            # correction for padded hts
            et = et.masked_fill(~enc_mask,-torch.inf)
            # calculate the alphas
            at = et.softmax(dim=-2)
            # calculate the context vector
            ct = (all_h_enc*at).sum(-2)
            emb_cur = emb[:,t,:]
            rt = torch.sigmoid(self.input_r(emb_cur) + self.hidden_r(s_prev) + self.context_r(ct))
            zt = torch.sigmoid(self.input_z(emb_cur) + self.hidden_z(s_prev) + self.context_z(ct))
            candidate_s = torch.tanh(self.input(emb_cur) + self.hidden(rt*s_prev) + self.context(ct))
            st = (1-zt)*candidate_s + zt*s_prev
            s_prev = st
            all_hidden[:,t,:] = st
        return all_hidden,st

    def evaluate(self,emb,all_h_enc,s_prev):
        with torch.no_grad():
            return self.forward(emb,all_h_enc,s_prev)
        
# %%

# decoder time

class Decoder(nn.Module):
    def __init__(self, V_t, E, H):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(V_t, E, max_norm=1, scale_grad_by_freq=True)
        self.attention = AttentionGRU(E,H)
        self.fc = nn.Linear(H,V_t)

        init.normal_(self.fc.weight,mean=0,std=0.1)
        
    def forward(self, x, all_hidden_enc):
        emb = self.embedding(x) # B,T,V 
        all_hidden, _ = self.attention(emb,all_hidden_enc) # B,T,H
        out = self.fc(all_hidden) # B,T,V
        return out
    
    def evaluate(self,x,all_hidden_enc,s_prev=None):
        with torch.no_grad():
            emb = self.embedding(x) # B,T,V but T will be 1
            all_h_dec, st = self.attention.evaluate(emb,all_hidden_enc,s_prev)
            out = self.fc(all_h_dec) # B,T,V but again T will be 1
            return out,st


# %%
class TranslationNN(nn.Module):
    def __init__(self,V_s,V_t,E,H):
        super(TranslationNN, self).__init__()
        self.encoder = Encoder(V_s,E,H)
        self.decoder = Decoder(V_t,E,H)

    def forward(self, x_s,x_t):
        all_h_enc = self.encoder(x_s)
        out = self.decoder(x_t,all_h_enc)
        return out
    
    def translate(self,x_s):
        # get all encoder representations
        all_h_enc = self.encoder.evaluate(x_s)
        B,T,H = all_h_enc.shape
        H = H//2 # Because we are using a bidirectional rnn in encoder
        x_t = torch.ones(B,1).long().to(device)
        x = torch.zeros(B,T).to(device) # making the wrong assumption that the max output has to be of size less than or equal to T_s, to be corrected later
        # start the process
        s_prev = torch.ones(B,H).to(device)
        counter = 0
        # end the process
        while (torch.all(torch.any(x==2,dim=0),dim=0).item()) or (counter <T):
            out,s_prev = self.decoder.evaluate(x_t,all_h_enc,s_prev) 
            probs = F.softmax(out,dim=-1)
            x_t = torch.argmax(probs,axis=-1)
            x[:,counter] = x_t.squeeze()
            counter+=1

        return x

# %%
# instantiate params
lr = learning_rate = 3e-4
model = TranslationNN(V_s,V_t,E,H)
model.to(device)
# load the weights from translation_model_v3.pth
# model.load_state_dict(torch.load('translation_model_v3.pth'))

loss_fn = nn.CrossEntropyLoss(reduction='none')
param_options = [
{'params': model.encoder.parameters(),'lr':3e-4},
{'params': model.decoder.embedding.parameters(),'lr':3e-4},
{'params': model.decoder.attention.parameters(),'lr':1e-3},
{'params':model.decoder.fc.parameters(),'lr':1e-3}
]
optim = AdamW(param_options,lr=lr)
# optim = AdamW(model.parameters(),lr=lr)

# make dataloaders

train_set, valid_set = random_split(dataset,[0.9,0.1])
train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=64)


for xs,xt,y in train_loader:
    xs,xt, y = xs.to(device),xt.to(device),y.to(device)
    break 
model.translate(xs)

# %%

epochs = 10


for epoch in range(epochs):
    train_loss, eval_loss = torch.zeros(len(train_loader)), torch.zeros(len(valid_loader))
    print(f"Epoch {epoch+1}/{epochs}")
    for c, (x_en,x_fr, y) in enumerate(tqdm(train_loader)):
        x_en,x_fr,y = x_en.to(device),x_fr.to(device),y.to(device)
        out = model(x_en,x_fr)
        out = out.permute(0,2,1)
        mask = (y!=0)
        loss = loss_fn(out,y)
        loss = loss[mask].mean()
        # backprop step
        optim.zero_grad()
        loss.backward()
        # clip the gradients
        clip_grad_norm_(model.parameters(),max_norm=1)
        # optimization step
        optim.step()
        train_loss[c] = loss
    # get the averaged training loss
    train_loss = train_loss.mean()
    print(f'Training Loss for Epoch {epoch+1} is {train_loss:.4f}')

    for c, (x_en,x_fr, y) in enumerate(valid_loader):
        with torch.no_grad():
            x_en, x_fr, y = x_en.to(device), x_fr.to(device), y.to(device)
            out = model(x_en,x_fr)
            out = out.permute(0,2,1)
            mask = (y!=0)
            loss = loss_fn(out,y)
            loss = loss[mask].mean()
            eval_loss[c] = loss

    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f'Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}')
# %%

    # BLEU score calculation
    detoken_en = MosesDetokenizer(lang='en')
    detoken_fr = MosesDetokenizer(lang='fr')
    scores = []
    with torch.no_grad():
        for x_en,x_fr,y in valid_loader:
            x_en,x_fr,y = x_en.to(device),x_fr.to(device),y.to(device)
            out = model.translate(x_en).long()
            # mask and get appropriate values
            mask = out == 2 
            correctmask = (mask.cumsum(dim=1)!=0)
            out[correctmask] = 0
            out_list = out.tolist()
            preds = [[dataset.kernel['target'].itos[i] for i in sublist if i !=0] for sublist in out_list]
            true_vals = [[dataset.kernel['target'].itos[i.item()] for i in sublist if i not in [0,1]] for sublist in x_fr]
            true_vals = [[i] for i in true_vals]
            score = bleu_score(preds,true_vals)
            scores.append(score)

    print(f'BLEU score is {torch.mean(torch.tensor(scores)):.4f}')
# %%

torch.save(model.state_dict(),'translation_model_v4.pth')