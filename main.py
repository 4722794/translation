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

param_options = [
    {'params':model.encoder.parameters(),'lr':3*config["lr"]},
    {'params':model.decoder.embedding.parameters(),'lr':3*config["lr"]},
    {'params':model.decoder.gru.parameters(),'lr':1e-4},
    {'params':model.decoder.attention.parameters(),'lr':config["lr"]},
    {'params':model.decoder.out.parameters(),'lr':0.5*config["lr"]},
    {'params':model.decoder.initialW}
]
optim = AdamW(param_options, lr=config["lr"])
loss_fn = nn.CrossEntropyLoss(reduction="none")

checkpoint = torch.load(checkpoint_path, map_location=device)
# load checkpoint details
model.load_state_dict(checkpoint["nn_state"])
optim.load_state_dict(checkpoint["opt_state"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

# %%
# make dataloaders
collate_fn = lambda x: (pad_sequence(i, batch_first=True) for i in x)
train_set, valid_set, test_set = random_split(
    dataset, [0.9, 0.05, 0.05], generator=torch.Generator().manual_seed(4722794)
)
train_sampler, val_sampler = BatchSampler(
    SubsetRandomSampler(train_set.indices),
    batch_size=config["batch_size"],
    drop_last=True,
), BatchSampler(
    SubsetRandomSampler(valid_set.indices),
    batch_size=config["batch_size"],
    drop_last=True,
)
test_sampler = BatchSampler(
    SubsetRandomSampler(test_set.indices),
    batch_size=config["batch_size"],
    drop_last=True,
)

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
test_loader = DataLoader(dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

# %%
# wandb section
wandb.login(key=api_key)
run = wandb.init(project="french", name="Baseline-lr scheduler", config=config)

wandb.watch(model, log_freq=100)

EOS_token = 2
num_epochs = config["epochs"]

for epoch in range(epoch, epoch + num_epochs):
    train_loss, eval_loss = torch.zeros(len(train_loader)), torch.zeros(len(val_loader))
    print(f"Epoch {epoch+1}")
    train_loss = train_loop(train_loader, model, optim, loss_fn, train_loss, device)
    # get the averaged training loss
    train_loss = train_loss.mean()
    print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")
    eval_loss = valid_loop(val_loader, model, loss_fn, eval_loss, inference_device)
    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}")

    if checkpoint_path.exists() and eval_loss < checkpoint["loss"]:
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = eval_loss
        checkpoint["nn_state"] = model.state_dict()
        checkpoint["opt_state"] = optim.state_dict()
        torch.save(checkpoint, checkpoint_path)

    scores = []
    for x_s, x_t, y in test_loader:
        with torch.no_grad():
            model.to(inference_device)
            x_s, x_t = x_s.to(inference_device), x_t.to(inference_device)
            outs, _ = model.evaluate(x_s)
        score = calculate_bleu_score(
            outs, x_t, dataset, EOS_token, device=inference_device
        )
        scores.append(score)

    mean_score = torch.tensor(scores).mean()
    print(f"Mean BLEU score is {mean_score:.4f}")
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": eval_loss,
        "bleu_score": mean_score,
    }
    run.log(metrics)
wandb.finish()

# %%
