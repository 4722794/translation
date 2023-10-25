#! python3
# %%
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from pathlib import Path
from scripts.dataset import (
    TranslationDataset,
)

import wandb
from dotenv import load_dotenv
import os,yaml
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler
load_dotenv()
from dataclasses import make_dataclass

api_key = os.getenv("WANDB_API_KEY")
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure paths
root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/checkpoint.pt")
train_path, val_path,test_path = data_path / "train/translations.csv", data_path / "val/translations.csv", data_path / "test/translations.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "tokenizer_en.model", data_path / "tokenizer_fr.model"
# get tokenizer
source_tokenizer,target_tokenizer = get_tokenizer(source_tokenizer_path), get_tokenizer(target_tokenizer_path)

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k,type(v)) for k,v in config.items()]
DotDict = make_dataclass('DotDict',fields)
c = DotDict(**config)

# %%
def init_checkpoint(config,checkpoint_path):
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
# %%
