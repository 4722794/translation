#!python3 utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# need attention maps

# attention stuff

def showAttention(input_words,output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # set up axes

    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



# bleu score

def calculate_bleu_score(outs, x_t, dataset, EOS_token):
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.long().tolist()
    preds = [dataset.sp_t.Decode(i).split() for i in out_list]
    targets = [dataset.sp_t.Decode(i) for i in x_t.to("cpu").long().tolist()]
    targets = [[i.split()] for i in targets]
    score = bleu_score(preds, targets)
    return 100*score

# basic translation

def token_to_sentence(outs,dataset,EOS_token):
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.tolist()
    preds = [dataset.sp_t.Decode(i) for i in out_list]
    return preds



def evaluate_show_attention(model,sentence,dataset,EOS_token):
    x_test = dataset.from_sentence_list('source',[sentence])
    outs,weights = model.evaluate(x_test)
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.long().tolist()
    out_list = [[i for i in sub_list if i !=0] for sub_list in out_list]
    preds = [ dataset.sp_t.IdToPiece(sublist + [2]) for sublist in out_list]
    original_sent = [dataset.sp_s.IdToPiece(sublist) for sublist in x_test.long().tolist()]
    # show attention maps
    showAttention(original_sent[0],preds[0],weights[0,:len(preds[0])])

    