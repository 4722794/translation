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
dataset = TranslationDataset(df,V_s,V_t,from_file=True)

#%%
E = 128
H = 256
B = 256


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