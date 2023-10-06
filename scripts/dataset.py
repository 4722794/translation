import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sacremoses import MosesTokenizer
import pandas as pd
from collections import namedtuple, Counter
from pathlib import Path
import ast
import numpy as np
import pickle


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

token_en = MosesTokenizer(lang='en')
token_fr = MosesTokenizer(lang='fr')

Kernel = namedtuple(f'Kernel',['vocab','stoi','itos','total_vocab'])

#%%

class TranslationDataset(Dataset):
    
    def __init__(self, path,v_source,v_target,from_file=False):
        super().__init__()
        self.kernel = {}
        self.X_s, self.X_t, self.Y = self.codex(path,v_source,v_target,from_file)

    def __len__(self):
        return len(self.X_s.data)

    def __getitem__(self, index):
        return self.X_s[index], self.X_t[index], self.Y[index]
    
    def codex(self,path,v_source,v_target,from_file):
        # read the file


        df = self.read_file(path)

        s_tokens = self.create_tokens(df.iloc[:,0],'source',v_source,token_en,from_file)
        X_source = pad_sequence([torch.tensor(seq) for seq in s_tokens],batch_first=True)
        
        t_tokens = self.create_tokens(df.iloc[:,1],'target',v_target,token_fr,from_file)
        X_target = pad_sequence([torch.tensor(seq[:-1]) for seq in t_tokens],batch_first=True)
        Y_target = pad_sequence([torch.tensor(seq[1:]) for seq in t_tokens],batch_first=True)

        return X_source,X_target,Y_target
    
    def create_tokens(self,data,datatype,vocab_size,tokenizer,from_file):

        if from_file:
            with open(f'data/{datatype}.pkl','rb') as f:
                self.kernel[f'{datatype}'] = kernel = pickle.load(f)
            itokens = pd.read_csv(f'data/{datatype}.csv', header=None)[0].apply(ast.literal_eval)
        else:
            self.kernel[f'{datatype}'] = kernel = self.create_vocab(vocab_size,data,tokenizer,datatype)
            if datatype.lower()=='source':
                data = data.apply(lambda x: x + ' EOS')
            else:
                data = data.apply(lambda x: 'BOS ' + x + ' EOS')
            # the tokenizer is currently from moses
            stokens = data.apply(lambda x: tokenizer.tokenize(x))
            itokens = stokens.apply(lambda x: [kernel.stoi.get(i,kernel.stoi['OOV']) for i in x])
            itokens.to_csv(f'data/{datatype}.csv',header=False,index=None)
        return itokens

    def create_vocab(self,vocab_size,s,tokenizer,datatype):
        """
        s: This is a pandas series object
        """
        # split it up
        s_words = s.apply(lambda x: tokenizer.tokenize(x))
        s_words = s_words.explode().value_counts()
        word_counts = Counter(s_words.to_dict())
        total_vocab = len(word_counts)
        vocab = [k for k,c in word_counts.most_common(vocab_size-4)]
        vocab+= ['OOV']
        stoi = {token:i for i,token in enumerate(vocab,3)}
        stoi['PAD'] = 0
        stoi['BOS'] = 1
        stoi['EOS'] = 2
        itos = {i:token for token,i in stoi.items()}
        k = Kernel(vocab,stoi,itos,total_vocab)
        with open(f'data/{datatype}.pkl','wb') as f:
            pickle.dump(k,f)
        return k

    def read_file(self,path):
        if isinstance(path,Path):
            df = pd.read_csv(path)
        elif isinstance(path,pd.DataFrame):
            df = path
        else:
            raise FileNotFoundError(f'{path} not a valid path or dataframe')
            
        
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
        s =  s.str.replace('[^a-zA-Z0-9àâäéèêëîïôöùûüÿç\-!\'?,. ]','',regex=True).str.lower()
        return s
#%%