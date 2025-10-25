#! python3
# %% imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from pathlib import Path
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationNN
from scripts.utils import token_to_sentence, train_loop, valid_loop,forward_pass,CustomAdam,save_checkpoint
from dotenv import load_dotenv
import os
import evaluate # this is a hugging face library
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler,get_bleu,init_checkpoint,check_resource_usage
import yaml
from dataclasses import make_dataclass
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from datetime import datetime
load_dotenv()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
logs_path = root_path / "logs"
logs_path.mkdir(exist_ok=True)

# Setup rich console and logging
log_file = logs_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
console = Console()

FORMAT = "%(message)s"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(FORMAT))

logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, markup=True, rich_tracebacks=True),
        file_handler
    ]
)
log = logging.getLogger("rich")

checkpoint_path = Path(f"{model_path}/checkpoint.tar")
train_path, val_path,test_path = data_path / "train/translations.csv", data_path / "val/translations.csv", data_path / "test/translations.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "tokenizer_en.model", data_path / "tokenizer_fr.model"

# optional, if you want to remove existing checkpoint
if checkpoint_path.exists():
    checkpoint_path.unlink()

#%% init tokenizer, checkpoint
source_tokenizer,target_tokenizer = get_tokenizer(source_tokenizer_path), get_tokenizer(target_tokenizer_path)

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k,type(v)) for k,v in config.items()]
DotDict = make_dataclass('DotDict',fields)
conf = DotDict(**config)

checkpoint = init_checkpoint(conf,checkpoint_path,device)
# check if checkpoint exists
if not checkpoint_path.exists():
    raise Exception("No checkpoint found.")

#%% main training loop
def main(config=None,checkpoint=None):
    log.info(f"[bold blue]Starting training on device: {device}[/]", extra={"markup": True})
    log.info(f"Configuration: {config}")

    # get dataset
    log.info("[yellow]Loading datasets...[/]", extra={"markup": True})
    train_set,val_set,test_set = get_dataset(train_path,source_tokenizer,target_tokenizer), get_dataset(val_path,source_tokenizer,target_tokenizer), get_dataset(test_path,source_tokenizer,target_tokenizer)
    log.info(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")

    # get loaders
    log.info("[yellow]Creating data loaders...[/]", extra={"markup": True})
    train_loader,val_loader,test_loader = get_dataloader(train_set,config['batch_size']), get_dataloader(val_set,config['batch_size']), get_dataloader(test_set,config['batch_size'])
    log.info(f"Batch size: {config['batch_size']}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # get model
    log.info("[yellow]Initializing model...[/]", extra={"markup": True})
    model = get_model(config['vocab_source'],config['vocab_target'],config['embedding_size'],config['hidden_size'],config['dropout'],config['dropout'],config['num_layers'],config['dot_product'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # get optimizer
    log.info("[yellow]Setting up optimizer and scheduler...[/]", extra={"markup": True})
    optim = get_optimizer(model, config['optimizer'], config['learning_rate'])
    scheduler = get_scheduler(optim, config['scheduler'])
    log.info(f"Optimizer: {config['optimizer']}, LR: {config['learning_rate']}, Scheduler: {config['scheduler']}")

    # loss fn
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # # Check resource usage and optimize batch size (optional, uncomment to use)
    # final_batch_size = check_resource_usage(model, train_loader, loss_fn, device, config['batch_size'])
    # if final_batch_size != config['batch_size']:
    #     config['batch_size'] = final_batch_size
    #     train_loader = get_dataloader(train_set, final_batch_size)
    #     val_loader = get_dataloader(val_set, final_batch_size)
    #     test_loader = get_dataloader(test_set, final_batch_size)
    #     log.info(f"[bold green]Updated batch size to: {final_batch_size}[/bold green]", extra={"markup": True})

    # training loop
    num_epochs = config['num_epochs']
    log.info(f"[bold green]Starting training for {num_epochs} epochs[/bold green]", extra={"markup": True})

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        # Create tasks for epoch, training, and validation
        epoch_task = progress.add_task("[yellow]Overall Progress", total=num_epochs)

        for epoch in range(num_epochs):
            log.info(f"[bold cyan]Epoch {epoch+1}/{num_epochs}[/bold cyan]", extra={"markup": True})

            # Create training task
            train_task = progress.add_task("[cyan]Training", total=len(train_loader))
            train_loss = train_loop(model, train_loader, loss_fn, optim, scheduler, epoch, device, progress, train_task)
            progress.remove_task(train_task)
            log.info(f"[green]Training Loss: {train_loss:.4f}[/green]", extra={"markup": True})

            # Create validation task
            val_task = progress.add_task("[blue]Validation", total=len(val_loader))
            val_loss = valid_loop(model, val_loader, loss_fn, device, progress, val_task)
            progress.remove_task(val_task)
            log.info(f"[blue]Validation Loss: {val_loss:.4f}[/blue]", extra={"markup": True})

            # if validation loss lower than checkpoint, save new weights
            if val_loss < checkpoint["loss"]:
                log.info(f"[bold yellow]New best model! Previous: {checkpoint['loss']:.4f}, New: {val_loss:.4f}[/]", extra={"markup": True})
                checkpoint = save_checkpoint(checkpoint_path, model, epoch,val_loss,optim,scheduler)
                log.info(f"[yellow]Checkpoint saved at epoch {epoch+1}[/]", extra={"markup": True})
            else:
                log.info(f"No improvement. Best loss: {checkpoint['loss']:.4f}")

            bleu_score_test = get_bleu(model, test_loader, device)
            bleu_score_val = get_bleu(model,val_loader,device)
            log.info(f"[magenta]Val BLEU: {bleu_score_val:.4f} | Test BLEU: {bleu_score_test:.4f}[/magenta]", extra={"markup": True})

            # Update epoch progress
            progress.update(epoch_task, advance=1)

    log.info(f"[bold green]{'='*60}[/]", extra={"markup": True})
    log.info(f"[bold green]Training complete![/]", extra={"markup": True})
    log.info(f"[bold green]Final Test BLEU: {bleu_score_test:.4f}[/]", extra={"markup": True})
    log.info(f"[bold green]Best Validation Loss: {checkpoint['loss']:.4f}[/]", extra={"markup": True})
    log.info(f"[bold green]{'='*60}[/]", extra={"markup": True})
    log.info(f"Log saved to: {log_file}")

# run the experiment

if __name__ == "__main__":
    main(config=config,checkpoint=checkpoint)