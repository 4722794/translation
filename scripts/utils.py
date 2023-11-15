#!python3 utils.py
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import wandb
from tqdm.auto import tqdm

# attention stuff

def showAttention(input_words, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    # set up axes

    ax.set_xticklabels([""] + input_words, rotation=90)
    ax.set_yticklabels([""] + output_words)

    # show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# bleu score

def calculate_bleu_score(outs, x_t, dataset, device):
    preds = outs.to(device)
    mask = preds == dataset.sp_t.eos_id()
    correctmask = mask.cumsum(dim=1) != 0
    preds.masked_fill_(correctmask, 0)
    out_list = preds.long().tolist()
    preds = [dataset.sp_t.Decode(i).split() for i in out_list]
    targets = [dataset.sp_t.Decode(i) for i in x_t.to("cpu").long().tolist()]
    targets = [[i.split()] for i in targets]
    score = bleu_score(preds, targets)
    return score


# basic translation


def token_to_sentence(outs,tokenizer):
    preds = outs.to("cpu")
    mask = preds == tokenizer.eos_id() # this should be 2
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.long().tolist()
    preds = [tokenizer.Decode(i) for i in out_list]
    return preds

def evaluate_show_attention(model, sentence, dataset):
    x_test = dataset.from_sentence_list("source", [sentence])
    outs, weights = model.evaluate(x_test)
    preds = outs.to("cpu")
    mask = preds == dataset.sp_t.eos_id()
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.long().tolist()
    out_list = [[i for i in sub_list if i != 0] for sub_list in out_list]
    preds = [dataset.sp_t.IdToPiece(sublist + [2]) for sublist in out_list]
    original_sent = [
        dataset.sp_s.IdToPiece(sublist) for sublist in x_test.long().tolist()
    ]
    # show attention maps
    showAttention(original_sent[0], preds[0], weights[0, : len(preds[0])])


# for training


def forward_pass(batch, model, loss_fn, device):
    x_s, x_t, y = batch
    x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)
    model.to(device)
    out = model(x_s, x_t)
    out = out.permute(0, 2, 1)
    mask = y != 0
    loss = loss_fn(out, y)
    loss = loss[mask].mean()
    return loss


def log_loss(loss,iteration,epoch,train=True):
    """
    log_loss: here log means "logging data" not "natural log"
    """
    prefix = 'train' if train else 'val'
    wandb.log({f"iterplots/{prefix}loss":loss.item(),"iterplots/iteration":iteration,"iterplots/epoch":epoch})
    if iteration % 100 == 0:
        print(f'Loss after {iteration} iterations is {loss.item():.4f}')


def train_loop(model,loader, loss_fn, optim, scheduler, epoch,device,log=True):
    model.to(device)
    model.train()
    loss_tensor = torch.zeros(len(loader))
    for c, batch in tqdm(enumerate(loader)):
        loss = forward_pass(batch, model, loss_fn, device)
        # backprop step
        optim.zero_grad()
        loss.backward()
        # optimization step
        optim.step()
        # scheduler step
        if scheduler is not None:
            scheduler.step(epoch + c / len(loader))
        iteration = len(loader)*epoch + c
        if log:
            log_loss(loss,iteration,epoch)
        loss_tensor[c] = loss.item()
    return loss_tensor.mean()

def valid_loop( model, loader,loss_fn, device):
    model.to(device)
    model.eval()
    loss_tensor = torch.zeros(len(loader))
    for c, batch in enumerate(loader):
        with torch.inference_mode():
            loss = forward_pass(batch, model, loss_fn, device)
        loss_tensor[c] = loss.item()
    return loss_tensor.mean()


# custom optimizer

# custom adam
class CustomAdam(AdamW):
    def __init__(self, *args, **kwargs):
        super(CustomAdam, self).__init__(*args, **kwargs)
        self.hooks = []

    def register_hook(self, hook):
        self.hooks.append(hook)

    def step(self, closure=None):
        # Store parameter values before update
        pre_params = {}
        for i, group in enumerate(self.param_groups):
            pre_params[i] = [p.clone() for p in group['params']]

        # Perform actual update
        super().step(closure)

        # Compute and save updates
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if len(p.shape)==2:
                    update = p.data - pre_params[i][j].data
                    self.hooks[0](update, pre_params[i][j].data)  # Assuming you have one common hook for all groups
# Define hook function to save updates


# custom learning scheduler

# Custom Scheduler Class
class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, freq_decay=0.5,amp_decay = 0.95, last_epoch=-1):
        self.freq_decay = freq_decay
        self.amp_decay = amp_decay
        super(CustomScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        omega = (1 + self.last_epoch) * self.freq_decay
        amplitude = self.amp_decay ** (self.last_epoch+1)
        return [lr*(1 + amplitude *np.sin(np.pi + omega)) for lr in self.base_lrs]



def save_checkpoint(checkpoint_path,model, epoch, loss, optimizer, scheduler,bleu_score):
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "bleu": bleu_score,
        "nn_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


#%%

    ### GARBAGE ZONE


def train(model,loader,val_loader,loss_fn,optimizer,scheduler,config,device):
    wandb.watch(model,loss_fn,log="all",log_freq=10)

    num_epochs = 20000/ len(loader) # 20000 rougly corresponds to 5 epochs if batch size is 512
    for epoch in range(num_epochs):
        for c,batch in enumerate(loader):
            loss = forward_pass(batch,model,loss_fn,device)
            # backprop step
            optimizer.zero_grad()
            loss.backward()
            # optimization step
            optimizer.step()
            # scheduler step
            if scheduler:
                scheduler.step(epoch + c/len(loader))
            iteration = len(loader)*epoch + c
            log_loss(loss,iteration,epoch)

        for c,batch in enumerate(val_loader):
            loss = forward_pass(batch,model,loss_fn,device)
            iteration = len(loader)*epoch + c
            log_loss(loss,iteration,epoch)
