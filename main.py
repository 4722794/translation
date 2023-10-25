#! python3
# %%
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
import wandb
from dotenv import load_dotenv
import os
import evaluate # this is a hugging face library
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler,get_bleu,init_checkpoint
import yaml
from dataclasses import make_dataclass
from tqdm.auto import tqdm
load_dotenv()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/checkpoint.tar")
train_path, val_path,test_path = data_path / "train/translations.csv", data_path / "val/translations.csv", data_path / "test/translations.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "tokenizer_en.model", data_path / "tokenizer_fr.model"

# optional, if you want to remove existing checkpoint
# if checkpoint_path.exists():
#     checkpoint_path.unlink()

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

#%%
def main(config=None,project=None,name=None):
    
    # keep the entire code within the wandb context manager

    with wandb.init(config=config,project=project,name=name):
        # all the code goes here
        c = wandb.config
        # get dataset
        train_set,val_set,test_set = get_dataset(train_path,source_tokenizer,target_tokenizer), get_dataset(val_path,source_tokenizer,target_tokenizer), get_dataset(test_path,source_tokenizer,target_tokenizer)
        # get loaders
        train_loader,val_loader,test_loader = get_dataloader(train_set,c.batch_size), get_dataloader(val_set,c.batch_size), get_dataloader(test_set,c.batch_size)
        # get model
        model = get_model(c.vocab_source,c.vocab_target,c.embedding_size,c.hidden_size,c.dropout,c.dropout,c.num_layers,c.dot_product)
        # get optimizer
        optim = get_optimizer(model, c.optimizer, c.learning_rate)
        # OPTIONAL: get_scheduler
        scheduler = get_scheduler(optim, c.scheduler)
        # loss fn
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        # training loop
        num_epochs = c.num_epochs # hard coding this value for now until further discussion
        for epoch in tqdm(range(num_epochs)):
            print(f"Epoch {epoch+1}")
            train_loss = train_loop(model, train_loader, loss_fn, optim,scheduler,epoch, device)
            print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")
            val_loss = valid_loop(model, val_loader, loss_fn, device)
            print(f"Validation Loss for Epoch {epoch+1} is {val_loss:.4f}")
            # if validation loss lower than checkpoint, save new weights
            if val_loss < checkpoint["loss"]:
                checkpoint = save_checkpoint(checkpoint_path, model, epoch,val_loss,optim,scheduler)
            bleu_score_test = get_bleu(model, test_loader, device)
            bleu_score_val = get_bleu(model,val_loader,device)
            print(f"Mean BLEU score is {bleu_score_val:.4f}")
            metrics = {
                "baseplots/epoch": epoch,
                "baseplots/train_loss": train_loss,
                "baseplots/val_loss": val_loss,
                "baseplots/bleu_score_val": bleu_score_val,
                "baseplots/bleu_score_test": bleu_score_test
            }
            wandb.log(metrics)
        wandb.log({"bleu":bleu_score_test})

# run the experiment
name = "GRU-6"
project_name="french"

main(config=config,project=project_name,name=name)