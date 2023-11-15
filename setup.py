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
import gc
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
    # use rmsprop
    elif optim_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
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
    EOS_token = token_t.eos_id()
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

def init_checkpoint(config,checkpoint_path,device):
    # instantiate params
    model = get_model(config.vocab_source, config.vocab_target, config.embedding_size, config.hidden_size, config.dropout, config.dropout, config.num_layers, config.dot_product)
    model.to(device)
    optim = get_optimizer(model, config.optimizer, config.learning_rate)
    scheduler = get_scheduler(optim, config.scheduler)
    checkpoint = {
        "nn_state": model.state_dict(),
        "opt_state": optim.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": 0,
        "loss": torch.inf,
        "bleu":0
    }
    torch.save(checkpoint, checkpoint_path)

    return checkpoint

import math
import math

def find_lr(model, optimizer,loss_fn,loader, init_value=1e-8, final_value=1., beta=0.98, device='cuda'):
    num = len(loader) - 1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []

    for batch in loader:
        batch_num += 1
        model.to(device)
        loss = forward_pass(batch, model, loss_fn, device)
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    # get the min log lr
    min_log_lr = log_lrs[losses.index(min(losses))]
    # convert to normal scale
    min_lr = 10**min_log_lr / 10
    return min_lr, log_lrs, losses


def get_min_lr(train_path, val_path, test_path, source_tokenizer, target_tokenizer, batch_size, vocab_source, vocab_target, embedding_size, hidden_size, dropout, num_layers, dot_product, optimizer, learning_rate, device):
    # get dataset
    train_set,val_set,test_set = get_dataset(train_path,source_tokenizer,target_tokenizer), get_dataset(val_path,source_tokenizer,target_tokenizer), get_dataset(test_path,source_tokenizer,target_tokenizer)
    # get loaders
    train_loader,val_loader,test_loader = get_dataloader(train_set,batch_size), get_dataloader(val_set,batch_size), get_dataloader(test_set,batch_size)
    # get model
    model = get_model(vocab_source,vocab_target,embedding_size,hidden_size,dropout,dropout,num_layers,dot_product)
    # get optimizer
    optim = get_optimizer(model, optimizer, learning_rate)
    # loss fn
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # get the learning rate
    min_lr, log_lrs, losses = find_lr(model,optim,loss_fn,train_loader,init_value = 1e-8, final_value=10,device=device)
    del model,optim
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return min_lr
