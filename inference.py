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
from scripts.utils import calculate_bleu_score, train_loop, valid_loop, evaluate_show_attention
import wandb
from dotenv import load_dotenv
import os


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/Baseline_checkpoint.pt")

config = dict(
    epochs=50,
    batch_size=512,
    vocab_source=5001,
    vocab_target=5001,
    embedding_size=256,
    hidden_size=256,
    device=device,
    lr=3e-4,
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
# check the loss ball park
eval_loss =  torch.zeros(len(val_loader))
eval_loss = valid_loop(val_loader, model, loss_fn, eval_loss, device)
# get the averaged validation loss
eval_loss = eval_loss.mean()
print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f}")


# %%
EOS_token =2
scores = []
for x_s, x_t, y in test_loader:
    with torch.no_grad():
        model.to(device)
        x_s, x_t = x_s.to(device), x_t.to(device)
        outs, _ = model.evaluate(x_s)
    score = calculate_bleu_score(
        outs, x_t, dataset, EOS_token, device=device
    )
    scores.append(score)
mean_score = torch.tensor(scores).mean()
print(f'BLEU score is {100*mean_score:.4f}')
# %%
# sample inference

source_sentences = [
    "Can you walk?",
    "What is love?",
    "I am very happy",
    "Can you give me a cheese omlette?",
    "I dream in french.",
    "I saw you cooking.",
    "He says he met my mother.",
    "Would you lend me some money?",
    "I'm looking forward to seeing you next Sunday.",
    "The war between the two countries ended with a big loss for both sides.",
    "The only thing that really matters is whether or not you did your best.",
    "The police compared the fingerprints on the gun with those on the door.",
    "The English language is undoubtedly the easiest and at the same time the most efficient means of international communication.",
    "I only eat the vegetables that I grow myself.",
]

x_test = dataset.from_sentence_list('source',source_sentences)

outs,weights = model.evaluate(x_test)

def token_to_sentence(outs,dataset,EOS_token):
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.long().tolist()
    preds = [dataset.sp_t.Decode(i) for i in out_list]
    return preds


preds = token_to_sentence(outs,dataset,EOS_token) 


# %%
eg_sent = source_sentences[0]


evaluate_show_attention(model,eg_sent,dataset,EOS_token)
#%%