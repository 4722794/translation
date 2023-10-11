#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sacremoses import MosesTokenizer, MosesDetokenizer
import pandas as pd
from pathlib import Path
import numpy as np
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationNN
from scripts.utils import calculate_bleu_score
from dotenv import load_dotenv
import os
from torchtext.data.metrics import bleu_score


# %%
device = torch.device("cpu")

token_en = MosesTokenizer(lang="en")
token_fr = MosesTokenizer(lang="fr")

df = pd.read_csv(
    "data/fra.txt", sep="\t", header=None, names=["en", "fr", "attribution"]
)
df.drop("attribution", axis=1, inplace=True)

checkpoint_path = Path("data/checkpoint.pt")

config = dict(
    epochs=50,
    batch_size=512,
    learning_rate=3e-4,
    vocab_source=2001,
    vocab_target=5001,
    embedding_size=128,
    hidden_size=64,
    device=device,
    lr=3e-4,
)
dataset = TranslationDataset(
    df, config["vocab_source"], config["vocab_target"], from_file=True
)

BOS_token = 1
EOS_token = 2

# %%
# instantiate params
model = TranslationNN(
    config["vocab_source"],
    config["vocab_target"],
    config["embedding_size"],
    config["hidden_size"],
)
model.to(device)
loss_fn = nn.CrossEntropyLoss(reduction="none")
checkpoint = torch.load(checkpoint_path,map_location=device)
# load checkpoint details
model.load_state_dict(checkpoint["nn_state"])


def collate_fn(batch):
    xs, xt = zip(*batch)
    x_s = [torch.Tensor(x) for x in xs]
    x_t = [torch.Tensor(x[:-1]) for x in xt]
    y = [torch.Tensor(x[1:]) for x in xt]
    x_s = pad_sequence(x_s, batch_first=True)
    x_t = pad_sequence(x_t, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x_s.long(), x_t.long(), y.long()


_,val_set, test_set = random_split(dataset, [0.9, 0.05,0.05])

#%%
val_loader = DataLoader(
    val_set, batch_size=config["batch_size"], collate_fn=collate_fn
)

test_loader = DataLoader(
    test_set, batch_size=config["batch_size"], collate_fn=collate_fn
)


#%%
# check the loss ball park
for x_s,x_t,y in val_loader:
    break
with torch.no_grad():
    out = model(x_s, x_t, device)
    out = out.permute(0, 2, 1)
    mask = y != 0
    loss = loss_fn(out, y)
    loss = loss[mask].mean()
# %%
scores = []
for x_s,x_t,y in test_loader:
    break
# outs,weights = model.eval(x_s)
# score = calculate_bleu_score(outs, x_t, dataset, EOS_token)
# scores.append(score)

# %%

detoken_en = MosesDetokenizer(lang="en")
detoken_fr = MosesDetokenizer(lang="fr")

preds = [detoken_fr.detokenize(p) for p in preds]
targets = [detoken_fr.detokenize(a) for a in targets]
# sentences = [detoken_en.detokenize(s) for s in sentences]


# %%
