#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.utils import calculate_bleu_score
from scripts.model import TranslationNN
import wandb
from dotenv import load_dotenv
import os
import random

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/checkpoint.pt")

config = dict(
    epochs=100,
    batch_size=256,
    vocab_source=5001,
    vocab_target=5001,
    embedding_size=256,
    hidden_size=256,
    device=device,
    lr=1e-4,
)

train_path, val_path, test_path = (
    data_path / "train/translations.csv",
    data_path / "valid/translations.csv",
    data_path / "test/translations.csv",
)
train_set,valid_set,test_set = (
    TranslationDataset(train_path),
    TranslationDataset(val_path),
    TranslationDataset(test_path),
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
optim = AdamW(model.parameters(), lr=config["lr"])
loss_fn = nn.CrossEntropyLoss(reduction="none")
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path,map_location=device)
    # load checkpoint details
    model.load_state_dict(checkpoint["nn_state"])
    optim.load_state_dict(checkpoint["opt_state"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
else:
    checkpoint = dict(
        epoch=0,
        loss=np.inf,
        nn_state=model.state_dict(),
        opt_state=optim.state_dict(),
    )
    torch.save(checkpoint, checkpoint_path)
# %%
# make dataloaders


def collate_fn(batch):
    xs, xt = zip(*batch)
    x_s = [torch.Tensor(x) for x in xs]
    x_t = [torch.Tensor(x[:-1]) for x in xt]
    y = [torch.Tensor(x[1:]) for x in xt]
    x_s = pad_sequence(x_s, batch_first=True)
    x_t = pad_sequence(x_t, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x_s.long(), x_t.long(), y.long()


train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    collate_fn=collate_fn,
    shuffle=True,)
valid_loader = DataLoader(
    valid_set,
    batch_size=config["batch_size"],
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_set,
    batch_size=config["batch_size"],
    collate_fn=collate_fn,
)

# %%

# wandb section
wandb.login(key=api_key)
run = wandb.init(project="français", name="Baseline", config=config)
log_table = wandb.Table(columns=["epoch", "train_loss", "val_loss"])
wandb.watch(model, log_freq=100)


if "epoch" not in locals():
    epoch = 0

num_epochs = config["epochs"]


for epoch in range(epoch, epoch + num_epochs):
    train_loss, eval_loss = torch.zeros(len(train_loader)), torch.zeros(
        len(valid_loader)
    )
    print(f"Epoch {epoch+1}")
    for c, (x_s, x_t, y) in enumerate(tqdm(train_loader)):
        x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)
        model.to(device)
        out = model(x_s, x_t, device)
        out = out.permute(0, 2, 1)
        mask = y != 0
        # check later if the mask is needed
        loss = loss_fn(out, y)
        loss = loss[mask].mean()
        # backprop step
        optim.zero_grad()
        loss.backward()
        # clip the gradients
        clip_grad_norm_(model.parameters(), max_norm=1)
        # optimization step
        optim.step()
        train_loss[c] = loss
    # get the averaged training loss
    train_loss = train_loss.mean()
    print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")

    for c, (x_s, x_t, y) in enumerate(valid_loader):
        with torch.no_grad():
            inference_device = torch.device("cpu")
            x_s, x_t, y = (
                x_s.to(inference_device),
                x_t.to(inference_device),
                y.to(inference_device),
            )
            model.to(inference_device)
            out = model(x_s, x_t, inference_device)
            out = out.permute(0, 2, 1)
            mask = y != 0
            loss = loss_fn(out, y)
            loss = loss[mask].mean()
            eval_loss[c] = loss

    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}")
    log_table.add_data(epoch + 1, train_loss, eval_loss)
    run.log(
        dict(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=eval_loss,
            training_log=log_table,
        )
    )
    if checkpoint_path.exists() and eval_loss < checkpoint["loss"]:
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = eval_loss
        checkpoint["nn_state"] = model.state_dict()
        checkpoint["opt_state"] = optim.state_dict()
        torch.save(checkpoint, checkpoint_path)
# %%
EOS_token = 2
scores = []
for x_s, x_t, y in test_loader:
    x_s, x_t = x_s.to(device), x_t.to(device)
    outs, weights = model.evaluate(x_s, device=device)
    score = calculate_bleu_score(outs, x_t, test_set, EOS_token)
    scores.append(score)

mean_score = torch.tensor(scores).mean()
#%%
wandb.finish()
