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
from scripts.utils import calculate_bleu_score, train_loop, valid_loop,forward_pass,CustomAdam,save_checkpoint
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
checkpoint_path = Path(f"{model_path}/checkpoint.tar")

config = dict(
    epochs=100,
    batch_size=512,
    vocab_source=5001,
    vocab_target=5001,
    embedding_size=256,
    hidden_size=256,
    device=device,
    lr=1e-3,
)

df = pd.read_csv(f"{data_path}/fra-eng.csv")
# only taking first 100 elements
# df = df.head(100)
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
    {'params':model.encoder.parameters(),'lr':0.5*config["lr"]},
    {'params':model.decoder.parameters()}]
optim = CustomAdam(param_options, lr=config["lr"])
#%%
def save_update(update, pre_param):
    # Do something with the update, like saving or logging it
    ratio = (update.std()/(pre_param.std() + 1e-8))
    update_data_ratio_batch.append(ratio.log10().item())

total_param_count = 0
for p in model.parameters():
    total_param_count+=1

optim.register_hook(save_update)
scheduler = CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2, eta_min=1e-5)
loss_fn = nn.CrossEntropyLoss(reduction="none")
if not checkpoint_path.exists():
    checkpoint = save_checkpoint(checkpoint_path, model,0,torch.inf, optim, scheduler)
else:
    checkpoint = torch.load(checkpoint_path)
# to save update/dataratio

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
run = wandb.init(project="french", name="Skip connections", config=config)

wandb.watch(model, log_freq=100)

EOS_token = 2
num_epochs = config["epochs"]
update_data_ratio_batch = []
keys = [f'batch/{n}' for n,p in model.named_parameters() if len(p.shape)==2]

def train_loop(loader, model, optim, scheduler, loss_fn, loss_tensor, epoch,device):
    model.to(device)
    model.train()
    for c, batch in enumerate(loader):
        loss = forward_pass(batch, model, loss_fn, device)
        # backprop step
        optim.zero_grad()
        loss.backward()
        # optimization step
        optim.step()
        # scheduler step
        scheduler.step(epoch + c / len(loader))
        loss_tensor[c] = loss.item()
        fine_metric = dict(zip(keys,update_data_ratio_batch))
        update_data_ratio_batch.clear()
        fine_metric['batch/iter'] = epoch*len(loader)+c
        run.log(fine_metric)
        
    return loss_tensor
#%%
for epoch in range(num_epochs):
    train_loss, eval_loss = torch.zeros(len(train_loader)), torch.zeros(len(val_loader))
    print(f"Epoch {epoch+1}")
    train_loss = train_loop(train_loader, model, optim, scheduler, loss_fn, train_loss, epoch,device)
    # get the averaged training loss
    train_loss = train_loss.mean()
    print(f"Training Loss for Epoch {epoch+1} is {train_loss:.4f}")
    eval_loss = valid_loop(val_loader, model, loss_fn, eval_loss, inference_device)
    # get the averaged validation loss
    eval_loss = eval_loss.mean()
    print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}")

    if eval_loss < checkpoint["loss"]:
        check_point = save_checkpoint(checkpoint_path, model,epoch,eval_loss, optim, scheduler)
        

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
        "baseplots/epoch": epoch,
        "baseplots/train_loss": train_loss,
        "baseplots/val_loss": eval_loss,
        "baseplots/bleu_score": mean_score,
    }
    run.log(metrics)
wandb.finish()

# %%
