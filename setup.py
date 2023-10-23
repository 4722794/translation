#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from pathlib import Path
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationDNN
from scripts.utils import token_to_sentence, train_loop, valid_loop,forward_pass,CustomAdam,save_checkpoint,CustomScheduler
import wandb
from dotenv import load_dotenv
import os
import evaluate # this is a hugging face library
import sentencepiece as spm
"""
config list
- Vs,Vt,E,H,

"""
#%%

# set up dataset 

def get_tokenizer(tokenizer_path):
    token = spm.SentencePieceProcessor()
    token.Load(str(tokenizer_path))
    return token

def get_dataset(df_path,token_s,token_t):
    df = pd.read_csv(df_path)
    dataset = TranslationDataset(df,token_s,token_t)
    return dataset

collate_fn = lambda x: (pad_sequence(i, batch_first=True) for i in x)

def get_dataloader(dataset,batch_size,shuffle=True):
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)
    return loader


# get model

def get_model(vocab_source,vocab_target,emb_size,hidden_size,dropout_encoder,dropout_decoder,num_layers,dot_product):
    model = TranslationDNN(V_s=vocab_source,V_t=vocab_target,E=emb_size,H=hidden_size,drop_e=dropout_encoder,drop_d=dropout_decoder,n=num_layers,dot=dot_product)
    return model

# get optimizer

def get_optimizer(model, optim_name, learning_rate):
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                               lr=learning_rate)
    else:
        raise ValueError("Optimizer not recognized")
    return optimizer

def get_scheduler(optimizer, scheduler_name):
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=10,
                                                T_mult=2,
                                                eta_min=1e-6)
    elif scheduler_name == "custom":
        scheduler = CustomScheduler(optimizer)
    else:
        raise ValueError("Scheduler not recognized")
    return scheduler


def get_bleu(model, test_loader, device):
    preds_list, actuals_list = list(), list()
    bleu = evaluate.load('bleu')
    token_t = test_loader.dataset.sp_t
    for x_s, x_t, _ in test_loader:
        with torch.no_grad():
            model.to(device)
            x_s, x_t = x_s.to(device), x_t.to(device)
            outs, _ = model.evaluate(x_s)
        preds = token_to_sentence(outs,token_t)
        actuals = token_to_sentence(x_t, token_t)
        preds_list.extend(preds)
        actuals_list.extend(actuals)
    predictions = preds_list
    references = [[i] for i in actuals_list]
    try:
        score = bleu.compute(predictions=predictions, references=references)['bleu']
    except ZeroDivisionError:
        score = 0
    return score

#%%