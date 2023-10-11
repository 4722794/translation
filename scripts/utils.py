#!python3 utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.data.metrics import bleu_score


# need attention maps


# bleu score


def calculate_bleu_score(outs, x_t, dataset, EOS_token):
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.tolist()
    preds = [
        [dataset.kernel["target"].itos[i] for i in sublist if i != 0]
        for sublist in out_list
    ]
    targets = [
        [dataset.kernel["target"].itos[i.item()] for i in sublist if i not in [0, 1]]
        for sublist in x_t.to("cpu")
    ]
    targets = [[i] for i in targets]
    score = bleu_score(preds, targets)
    return score


# basic translation
