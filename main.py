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
from scripts.model import Encoder,Decoder
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

token_en = MosesTokenizer(lang='en')
token_fr = MosesTokenizer(lang='fr')

df = pd.read_csv('data/fra.txt',sep='\t',header=None,names=['en','fr','attribution'])
df.drop('attribution',axis=1,inplace=True)

checkpoint_path = Path('data/checkpoint.pt')

V_s = vocab_source = 2001
V_t = vocab_target = 5001
E = 128
H = 256
B = 256
dataset = TranslationDataset(df,V_s,V_t,from_file=True)

#%%

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

# %%
# instantiate params
lr = learning_rate = 3e-4
model = TranslationNN(V_s,V_t,E,H)
optim = AdamW(model.parameters(),lr=lr)
loss_fn = nn.CrossEntropyLoss(reduction='none')
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    # load checkpoint details 
    model.load_state_dict(checkpoint['nn_state'])
    optim.load_state_dict(checkpoint['opt_state'])
    epoch = checkpoint['epoch']
    loss=checkpoint['loss']

# make dataloaders

def collate_fn(batch):
    xs, xt = zip(*batch)
    xs = [torch.Tensor(x) for x in xs]
    xt = [torch.Tensor(x[:-1]) for x in xt]
    y = [torch.Tensor(x[1:]) for x in xt]
    xs = pad_sequence(xs, batch_first=True)
    xt = pad_sequence(xt, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return xs.long(), xt.long(), y.long()

train_set, valid_set = random_split(dataset,[0.9,0.1])
train_loader = DataLoader(train_set,batch_size=B,collate_fn=collate_fn,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=B,collate_fn=collate_fn)

# %%
if 'epoch' not in locals():
    epoch = 0
num_epochs = 1


for epoch in range(epoch,epoch+num_epochs):
    train_loss, eval_loss = torch.zeros(len(train_loader)), torch.zeros(len(valid_loader))
    print(f"Epoch {epoch+1}")
    for c, (x_s,x_t, y) in enumerate(tqdm(train_loader)):
        x_s,x_t,y = x_s.to(device),x_t.to(device),y.to(device)
        model.to(device)
        out = model(x_s,x_t)
        out = out.permute(0,2,1)
        mask = (y!=0)
        # check later if the mask is needed
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

    for c, (x_s,x_t, y) in enumerate(valid_loader):
        with torch.no_grad():
            device = torch.device('cpu')
            x_s, x_t,y = x_s.to(device), x_t.to(device), y.to(device)
            model.to(device)
            out = model(x_s,x_t)
            out = out.permute(0,2,1)
            mask = (y!=0)
            loss = loss_fn(out,y)
            loss = loss[mask].mean()
            eval_loss[c] = loss

    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f'Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}')
    if checkpoint_path.exists() and eval_loss < checkpoint['loss']:
        checkpoint['epoch'] = epoch
        checkpoint['loss'] = eval_loss
        checkpoint['nn_state'] = model.state_dict()
        checkpoint['opt_state'] = optim.state_dict()
        torch.save(checkpoint,checkpoint_path)
# %%
