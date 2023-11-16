#! python3
# %%
import torch
import torch.nn as nn
from pathlib import Path
from scripts.utils import train_loop, valid_loop, save_checkpoint
import wandb
from dotenv import load_dotenv
from setup import get_tokenizer,get_dataset,get_dataloader,get_model,get_optimizer,get_scheduler,get_bleu,init_checkpoint, get_min_lr
import yaml
from dataclasses import make_dataclass
from tqdm.auto import tqdm
load_dotenv()
################################################# CODE BELOW #################################################
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
model_path = root_path / "saved_models"
temp_path = root_path / "temp"
data_path = temp_path / "data"
checkpoint_path = Path(f"{model_path}/checkpoint.tar")
train_path, val_path,test_path = data_path / "train/returnleg.csv", data_path / "val/returnleg.csv", data_path / "test/returnleg.csv"
source_tokenizer_path, target_tokenizer_path = data_path / "en.model", data_path / "sp.model"

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k,type(v)) for k,v in config.items()]
DotDict = make_dataclass('DotDict',fields)
conf = DotDict(**config)

#%%
source_tokenizer,target_tokenizer = get_tokenizer(source_tokenizer_path), get_tokenizer(target_tokenizer_path)

# create checkpoint

# optional, if you want to remove existing checkpoint
if checkpoint_path.exists():
    checkpoint_path.unlink()

checkpoint = init_checkpoint(conf,checkpoint_path,device)
# check if checkpoint exists
if not checkpoint_path.exists():
    raise Exception("No checkpoint found.")

#%%
def main(config=None,project=None,name=None,checkpoint=None):
    
    # keep the entire code within the wandb context manager

    with wandb.init(config=config,project=project,name=name) as run:
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
            bleu_score_val = get_bleu(model,val_loader,device)
            print(f"Mean BLEU score is {bleu_score_val:.4f}")
            # if BLEU score improves, only then save model
            if bleu_score_val > checkpoint["bleu"]:
                checkpoint = save_checkpoint(checkpoint_path, model, epoch,val_loss,optim,scheduler,bleu_score_val,run)
            bleu_score_test = get_bleu(model, test_loader, device)
            metrics = {
                "Training Log/Epoch":epoch,
                "Training Log/Train Loss":train_loss,
                "Training Log/Teach Val Loss":val_loss,                
                "baseplots/epoch": epoch,
                "baseplots/train_loss": train_loss,
                "baseplots/val_loss": val_loss,
                "baseplots/bleu_score_val": bleu_score_val,
                "baseplots/bleu_score": bleu_score_test
            }
            wandb.log(metrics)
        wandb.log({"bleu":bleu_score_test})

main(config=conf,project="ucl_return",name="PSGRU-Camp Nou",checkpoint=checkpoint)