#%%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import pandas as pd
from collections import namedtuple, Counter
from pathlib import Path
import ast
import numpy as np
import pickle
import sentencepiece as spm

#%%

class TranslationDataset(Dataset):
    
    def __init__(self,df,token_s,token_t):
        super().__init__()
        self.sp_s, self.sp_t = token_s, token_t
        self.sp_s.SetEncodeExtraOptions('eos')
        self.sp_t.SetEncodeExtraOptions('bos:eos')
        df = self.read_file(df)
        self.X_s, self.X_t,self.Y = self.codex(df)

    def __len__(self):
        return len(self.X_s.data)

    def __getitem__(self, index):
        return self.X_s[index], self.X_t[index], self.Y[index]
    
    def __getitems__(self,idx):
        return self.X_s[idx], self.X_t[idx],self.Y[idx]
    
    def codex(self,df):
       
        s_tokens = df.iloc[:,0].apply(lambda x: self.sp_s.EncodeAsIds(x))
        t_tokens = df.iloc[:,1].apply(lambda x: self.sp_t.EncodeAsIds(x))

        x_s,x_t, y = self.series_to_tensor(s_tokens,t_tokens)
        return x_s,x_t,y
    

    def series_to_tensor(self,s_tokens,t_tokens):
        x_s = s_tokens.apply(lambda x: torch.tensor(x,dtype=torch.long))
        x_t = t_tokens.apply(lambda x: torch.tensor(x[:-1],dtype=torch.long))
        y = t_tokens.apply(lambda x: torch.tensor(x[1:],dtype=torch.long))
        return x_s.values,x_t.values,y.values


    def read_file(self,df):
        df.iloc[:,0] = self.preprocess(df.iloc[:,0])
        df.iloc[:,1] = self.preprocess(df.iloc[:,1])
        df.replace(r"^\s*$",np.nan,regex=True,inplace=True)
        df.dropna(inplace=True)
        valid_sentences = df.iloc[:,1].str.split().apply(lambda x: len(x)) <=100
        return df[valid_sentences]
    
    def preprocess(self,s):
        """
        Whatever string there is, it must be preprocessed before it passes into any form of tokenization
        """
        if not isinstance(s,pd.Series):
            s = pd.Series(s)
        s = s.str.replace('\u202f|\xa0',' ',regex=False).str.lower()
        # s =  s.str.replace('[^a-zA-Z0-9àâäéèêëîïôöùûüÿç\-!\'?,. ]','',regex=True).str.lower()
        return s
    
    def from_sentence(self,datatype,sentence):

        tokenizer = self.sp_s if datatype=='source' else self.sp_t
        tokens = tokenizer.EncodeAsIds(sentence.lower())
        return torch.Tensor(tokens).long().to(device)
    
    def from_sentence_list(self,datatype,sentlist):
        itokens = [self.from_sentence(datatype,s) for s in sentlist]
        return pad_sequence(itokens,batch_first=True)
        

#%%
if __name__ == '__main__':
    root_path = Path(__file__).resolve().parents[1]
    temp_path = root_path / "temp"
    data_path = temp_path / 'data'
    s_tokenizer_path = data_path / 'tokenizer_en.model'
    t_tokenizer_path = data_path / 'tokenizer_fr.model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(f'{data_path}/fra-eng.csv')
    token_s,token_t = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
    token_s.Load(str(s_tokenizer_path))
    token_t.Load(str(t_tokenizer_path))    
    dataset = TranslationDataset(df,token_s,token_t)
    
# %%

