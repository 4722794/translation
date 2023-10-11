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
    ax.set_yticklabels(['']+output_words)

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
    out_list = preds.tolist()
    preds = [
        [dataset.kernel["target"].itos[i] for i in sublist if i != 0]
        for sublist in out_list
    ]
    targets = [
        [dataset.kernel["target"].itos[i.item()] for i in sublist if i not in [0, 1]]
        for sublist in x_t.to("cpu")
    ]
    targets = [[i] for i in targets]
    score = bleu_score(preds, targets)
    return score


# basic translation

def token_to_sentence(outs,dataset,EOS_token,detokenizer):
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.tolist()
    preds = [
        [dataset.kernel["target"].itos[i] for i in sublist if i != 0]
        for sublist in out_list
    ]
    # detokenize
    preds = [detokenizer.detokenize(i) for i in preds]
    return preds

def evaluate_show_attention(model,sentence,dataset,EOS_token):
    x_test = dataset.from_sentence_list('source',[sentence])
    outs,weights = model.evaluate(x_test)
    preds = outs.to("cpu")
    mask = preds == EOS_token
    correctmask = mask.cumsum(dim=1) != 0
    preds[correctmask] = 0
    out_list = preds.tolist()
    preds = [
        [dataset.kernel["target"].itos[i] for i in sublist if i != 0]+['EOS']
        for sublist in out_list
    ]
    original_sent = [dataset.kernel["source"].itos[i.item()] for i in x_test[0]]
    # show attention maps
    showAttention(original_sent,preds[0],weights[0,:len(preds[0])])

    