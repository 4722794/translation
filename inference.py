#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torchtext.data.metrics import bleu_score
from sacremoses import MosesTokenizer, MosesDetokenizer
import pandas as pd
from pathlib import Path
import numpy as np
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationNN
from dotenv import load_dotenv
import os


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


# %%
# instantiate params
model = TranslationNN(
    config["vocab_source"],
    config["vocab_target"],
    config["embedding_size"],
    config["hidden_size"],
)
model.to(device)

checkpoint = torch.load(checkpoint_path)
# load checkpoint details
model.load_state_dict(checkpoint["nn_state"], map_location=device)


def collate_fn(batch):
    xs, xt = zip(*batch)
    x_s = [torch.Tensor(x) for x in xs]
    x_t = [torch.Tensor(x[:-1]) for x in xt]
    y = [torch.Tensor(x[1:]) for x in xt]
    x_s = pad_sequence(x_s, batch_first=True)
    x_t = pad_sequence(x_t, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x_s.long(), x_t.long(), y.long()


train_set, valid_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_set, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True
)
valid_loader = DataLoader(
    valid_set, batch_size=config["batch_size"], collate_fn=collate_fn
)
