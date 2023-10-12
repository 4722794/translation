#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split,SubsetRandomSampler,BatchSampler
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
from scripts.model import TranslationNN
from scripts.utils import calculate_bleu_score
import wandb
from dotenv import load_dotenv
import os

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
    learning_rate=3e-4,
    vocab_source=5001,
    vocab_target=5001,
    embedding_size=256,
    hidden_size=256,
    device=device,
    lr=1e-4,
)

df = pd.read_csv(f"{data_path}/fra-eng.csv")
dataset = TranslationDataset(df, from_file=True)

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

collate_fn = lambda x: (pad_sequence(i,batch_first=True) for i in x)
    
train_set, valid_set, test_set = random_split(dataset, [0.9, 0.05, 0.05],generator=torch.Generator().manual_seed(4722794))

train_sampler, val_sampler = BatchSampler(SubsetRandomSampler(train_set.indices),batch_size=config["batch_size"],drop_last=True), BatchSampler(SubsetRandomSampler(valid_set.indices),batch_size=config["batch_size"],drop_last=True)
test_sampler = BatchSampler(SubsetRandomSampler(test_set.indices),batch_size=config["batch_size"],drop_last=True)

train_loader = DataLoader(
    dataset,
    batch_sampler=train_sampler,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset,
    batch_sampler=val_sampler,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    dataset,
    batch_sampler=test_sampler,
    collate_fn=collate_fn)

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
        len(val_loader)
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
        train_loss[c] = loss.item()
    # get the averaged training loss
    train_loss = train_loss.mean()
    print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")

    for c, (x_s, x_t, y) in enumerate(val_loader):
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
            eval_loss[c] = loss.item()

    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}")

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
    score = calculate_bleu_score(outs, x_t, dataset, EOS_token)
    scores.append(score)

mean_score = torch.tensor(scores).mean()
print(f"Mean BLEU score is {mean_score:.4f}")
wandb.finish()
