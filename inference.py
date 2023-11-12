#! python3
# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler

from torch.nn.utils import clip_grad_norm_
import pandas as pd
from pathlib import Path
from scripts.dataset import (
    TranslationDataset,
)  # The logic of TranslationDataset is defined in the file dataset.py
from scripts.model import TranslationNN
from scripts.utils import calculate_bleu_score, valid_loop, evaluate_show_attention,token_to_sentence
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler,init_checkpoint,get_bleu
import yaml
from dataclasses import make_dataclass
import evaluate

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = Path(f"{model_path}/best_run.tar")
train_path, val_path,test_path = data_path / "train/translations.csv", data_path / "val/translations.csv", data_path / "test/translations.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "tokenizer_en.model", data_path / "tokenizer_fr.model"


#%% init tokenizer, checkpoint
source_tokenizer,target_tokenizer = get_tokenizer(source_tokenizer_path), get_tokenizer(target_tokenizer_path)

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k,type(v)) for k,v in config.items()]
DotDict = make_dataclass('DotDict',fields)
c = DotDict(**config)

# check if checkpoint exists
if not checkpoint_path.exists():
    raise Exception("No checkpoint found.")

# %%
train_set,val_set,test_set = get_dataset(train_path,source_tokenizer,target_tokenizer), get_dataset(val_path,source_tokenizer,target_tokenizer), get_dataset(test_path,source_tokenizer,target_tokenizer)
# get loaders
train_loader,val_loader,test_loader = get_dataloader(train_set,c.batch_size), get_dataloader(val_set,c.batch_size), get_dataloader(test_set,c.batch_size)
# get model
model = get_model(c.vocab_source,c.vocab_target,c.embedding_size,c.hidden_size,c.dropout,c.dropout,c.num_layers,c.dot_product)

loss_fn = nn.CrossEntropyLoss(reduction="none")

checkpoint = torch.load(checkpoint_path, map_location=device)
# load checkpoint details
model.load_state_dict(checkpoint["nn_state"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
print(f"Loss after {epoch+1} epochs is {loss:.4f}")

# %%
# check the loss ball park
eval_loss = val_loss = valid_loop(model, val_loader, loss_fn, device)
# get the averaged validation loss
eval_loss = eval_loss.mean()
print(f"Validation Loss for Epoch {epoch+1} is {eval_loss:.4f} \n stored value is {loss:.4f}")

#%%
# %%
bleu_score_test = get_bleu(model, test_loader, device)
print(f'BLEU score is {100*bleu_score_test:.4f}')
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
#%%
# x_test = train_set.from_sentence_list('source',source_sentences)

# outs,weights = model.evaluate(x_test)


# preds = token_to_sentence(outs,train_set) 

# # %%
# eg_sent = source_sentences[-1]


# evaluate_show_attention(model,eg_sent,dataset,EOS_token)
#%%

# save files for measure
bleu = evaluate.load("bleu") 
pred_path = root_path/'machinetranslated.txt'
groundtruth = root_path/'groundtruth.txt'
preds_list = []
actuals_list = []

for xs,xt,_ in test_loader:
    with torch.no_grad():
        model.to(device)
        xt,xs = xt.to(device),xs.to(device)
        outs, _ = model.evaluate(xs)
    preds = token_to_sentence(outs,target_tokenizer)
    preds_list.extend(preds)
    actuals = token_to_sentence(xt,target_tokenizer)
    actuals_list.extend(actuals)

with open(pred_path,'w') as f:
    f.writelines('\n'.join(preds_list))

with open(groundtruth,'w') as f:
    f.writelines('\n'.join(actuals_list))

# %%
predictions = preds_list
references = [[i] for i in actuals_list]
# %
score = bleu.compute(predictions=predictions,references=references)
print(f"BLEU score is {100*score['bleu']:.4f}")
# %%