#!python3 
#%%
from pathlib import Path
import pandas as pd     
import sentencepiece as spm
#%%     

root_path = Path(__file__).resolve().parents[1]

data_path = root_path / 'data'
df = pd.read_csv(data_path / 'fra-eng.csv')
# store english only
df['en'].str.lower().to_csv(data_path / 'en.txt',index=False,header=None)
# store french only - remember the whole thing is lowercase
df['fr'].str.lower().to_csv(data_path / 'fr.txt',index=False,header=None)
#%%
# for training a tokenizer model
spm.SentencePieceTrainer.Train(f"--input={data_path}/en.txt --model_prefix={data_path}/tokenizer_en --vocab_size=5001 \
--unk_id=5000 --bos_id=1 --eos_id=2 --pad_id=0 \
--model_type=bpe")

spm.SentencePieceTrainer.Train(f"--input={data_path}/fr.txt --model_prefix={data_path}/tokenizer_fr --vocab_size=5001 \
--unk_id=5000 --bos_id=1 --eos_id=2 --pad_id=0\
--model_type=bpe")

# %%