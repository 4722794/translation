#! python3
# %%
import torch
import torch.nn as nn
from pathlib import Path
from scripts.utils import train_loop, valid_loop, save_checkpoint
import wandb
from dotenv import load_dotenv
import os
import evaluate # this is a hugging face library
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler,get_bleu,init_checkpoint, find_lr, get_min_lr
import yaml
from dataclasses import make_dataclass
from tqdm.auto import tqdm
load_dotenv()
################################################# CODE BELOW #################################################
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/checkpoint.tar")
train_path, val_path,test_path = data_path / "train/translation.csv", data_path / "val/translation.csv", data_path / "test/translation.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "tokenizer_en.model", data_path / "tokenizer_fr.model"

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k,type(v)) for k,v in config.items()]
DotDict = make_dataclass('DotDict',fields)
conf = DotDict(**config)

#%%
source_tokenizer,target_tokenizer = get_tokenizer(source_tokenizer_path), get_tokenizer(target_tokenizer_path)

#%%
def main(config=None,project=None,name=None,checkpoint=None):
    
    # keep the entire code within the wandb context manager

    with wandb.init(config=config,project=project,name=name):
        c = wandb.config
        # find learning rate
        # lr finder
        sample_lr = 0.001 # this doesn't even get used
        min_lr = get_min_lr(train_path, val_path, test_path, source_tokenizer, target_tokenizer, c.batch_size, c.vocab_source, c.vocab_target, c.embedding_size, c.hidden_size, c.dropout, c.num_layers, c.dot_product, c.optimizer, sample_lr, device)
        print(f"Minimum learning rate is {min_lr}")
        if min_lr < 1e-4:
            min_lr = 1e-4
        # get dataset
        train_set,val_set,test_set = get_dataset(train_path,source_tokenizer,target_tokenizer), get_dataset(val_path,source_tokenizer,target_tokenizer), get_dataset(test_path,source_tokenizer,target_tokenizer)
        # get loaders
        val_batch = 8
        train_loader,val_loader,test_loader = get_dataloader(train_set,c.batch_size), get_dataloader(val_set,val_batch), get_dataloader(test_set,val_batch)
        # get model
        model = get_model(c.vocab_source,c.vocab_target,c.embedding_size,c.hidden_size,c.dropout,c.dropout,c.num_layers,c.dot_product)
        # get optimizer
        optim = get_optimizer(model, c.optimizer, min_lr)
        # loss fn
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        # OPTIONAL: get_scheduler
        scheduler = get_scheduler(optim, c.scheduler)

        # training loop
        num_epochs = c.num_epochs # hard coding this value for now until further discussion
        for epoch in tqdm(range(num_epochs)):
            print(f"Epoch {epoch+1}")
            train_loss = train_loop(model, train_loader, loss_fn, optim,scheduler,epoch, device)
            print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")
            val_loss = valid_loop(model, val_loader, loss_fn, device)
            print(f"Validation Loss for Epoch {epoch+1} is {val_loss:.4f}")
            bleu_score = get_bleu(model, test_loader, device)
            print(f"Mean BLEU score is {bleu_score:.4f}")
            metrics = {
                "baseplots/epoch": epoch,
                "baseplots/train_loss": train_loss,
                "baseplots/val_loss": val_loss,
                "baseplots/bleu_score": bleu_score,
            }
            wandb.log(metrics)
        wandb.log({"bleu":bleu_score})

wandb.agent("y7ahqzp6",main,count=20,project="sweepstakes")
# main(config=conf,project="sweepstakes",name="sweetrun",checkpoint=checkpoint_path)
