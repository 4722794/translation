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
from scripts.model import TranslationNN, TranslationDNN
from scripts.utils import token_to_sentence, train_loop, valid_loop,forward_pass,CustomAdam,save_checkpoint,CustomScheduler
from dotenv import load_dotenv
import os
import evaluate # this is a hugging face library
import sentencepiece as spm
import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Confirm, IntPrompt
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

def get_model(vocab_source,vocab_target,emb_size,hidden_size,dropout_encoder,dropout_decoder,num_layers=None,dot_product=None):
    """
    Get translation model. If num_layers is provided, returns TranslationDNN (multi-layer).
    Otherwise returns TranslationNN (single-layer).
    """
    if num_layers is not None and num_layers > 1:
        # Use deep model (TranslationDNN)
        if dot_product is None:
            dot_product = False
        model = TranslationDNN(V_s=vocab_source,V_t=vocab_target,E=emb_size,H=hidden_size,
                               drop_e=dropout_encoder,drop_d=dropout_decoder,n=num_layers,dot=dot_product)
    else:
        # Use simple model (TranslationNN) - single layer, no dot product option
        model = TranslationNN(V_s=vocab_source,V_t=vocab_target,E=emb_size,H=hidden_size,
                             drop_e=dropout_encoder,drop_d=dropout_decoder)
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
        # Try using evaluate library first
        bleu = evaluate.load('bleu')
        score = bleu.compute(predictions=predictions, references=references)['bleu']
    except (FileNotFoundError, Exception):
        # Fallback: use sacrebleu directly
        try:
            import sacrebleu
            score = sacrebleu.corpus_bleu(predictions, [actuals_list]).score / 100.0
        except:
            # Last resort: return 0
            score = 0.0

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
    }
    torch.save(checkpoint, checkpoint_path)

    return checkpoint

def check_resource_usage(model, train_loader, loss_fn, device, initial_batch_size):
    """
    Check resource usage (GPU/CPU memory) with current batch size by running forward+backward pass.
    Asks user if they want to increase batch size.
    Returns the final batch size to use.
    """
    console = Console()
    current_batch_size = initial_batch_size

    while True:
        console.print(f"\n[bold cyan]Testing batch size: {current_batch_size}[/bold cyan]")

        # Get a sample batch
        sample_batch = next(iter(train_loader))
        x_s, x_t, _ = sample_batch
        x_s, x_t = x_s.to(device), x_t.to(device)

        # Run a full forward + backward pass to measure real memory usage
        model.train()
        try:
            # Clear cache before test
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            # Forward pass
            output = model(x_s, x_t)
            B, T, V = output.shape
            loss = loss_fn(output.reshape(-1, V), x_t.reshape(-1))
            mask = (x_t != 0).reshape(-1)
            loss = (loss * mask).sum() / mask.sum()

            # Backward pass
            loss.backward()

            # Get memory stats after forward+backward
            if device.type == 'cuda':
                mem_used = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
                mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                mem_type = "GPU"
            else:
                mem_info = psutil.virtual_memory()
                mem_used = mem_info.used / 1024**3  # GB
                mem_total = mem_info.total / 1024**3  # GB
                mem_type = "CPU"

            mem_percent = (mem_used / mem_total) * 100

            # Display memory usage
            console.print(f"{mem_type} Memory: {mem_used:.2f} GB / {mem_total:.2f} GB ({mem_percent:.1f}%)")

            # Ask if user wants to increase batch size
            if mem_percent < 80:  # Only suggest increase if under 80%
                if Confirm.ask(f"Memory at {mem_percent:.1f}%. Try larger batch size?", default=False):
                    new_batch_size = IntPrompt.ask(
                        "Enter new batch size",
                        default=current_batch_size * 2
                    )
                    current_batch_size = new_batch_size
                    # Recreate loader with new batch size
                    train_loader = get_dataloader(train_loader.dataset, current_batch_size)

                    # Clear cache
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    model.zero_grad()
                    continue
                else:
                    console.print(f"[bold green]Using batch size: {current_batch_size}[/bold green]")
                    model.zero_grad()
                    break
            else:
                console.print(f"[yellow]Memory usage high ({mem_percent:.1f}%). Keeping batch size at {current_batch_size}[/yellow]")
                model.zero_grad()
                break

        except RuntimeError as e:
            if "out of memory" in str(e):
                console.print(f"[bold red]OOM with batch size {current_batch_size}![/bold red]")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                new_batch_size = IntPrompt.ask(
                    "Enter smaller batch size",
                    default=max(1, current_batch_size // 2)
                )
                current_batch_size = new_batch_size
                train_loader = get_dataloader(train_loader.dataset, current_batch_size)
                model.zero_grad()
                continue
            else:
                raise e

    return current_batch_size