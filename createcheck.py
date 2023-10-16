#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler

from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from pathlib import Path
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationNN
from scripts.utils import calculate_bleu_score, train_loop, valid_loop
import wandb
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_device = torch.device("cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/checkpoint.pt")

config = dict(
    epochs=100,
    batch_size=512,
    vocab_source=5001,
    vocab_target=5001,
    embedding_size=256,
    hidden_size=256,
    device=device,
    lr=1e-4,
)

df = pd.read_csv(f"{data_path}/fra-eng.csv")
# only taking first 100 elements
dataset = TranslationDataset(df, from_file=False)

# %%
# instantiate params
model = TranslationNN(
    config["vocab_source"],
    config["vocab_target"],
    config["embedding_size"],
    config["hidden_size"],
)
model.to(device)
optim = AdamW(model.parameters(), lr=config["lr"])
loss_fn = nn.CrossEntropyLoss(reduction="none")

#%%
checkpoint = {
    "nn_state": model.state_dict(),
    "opt_state": optim.state_dict(),
    "epoch": 0,
    "loss": torch.inf,
}

torch.save(checkpoint, checkpoint_path)
# %%
