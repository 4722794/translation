#%%
import torch
from model import MultiDot

def test_multi_dot():
    hidden_size = 64
    num_heads = 8
    batch_size = 16
    seq_len = 10

    multi_dot = MultiDot(hidden_size, num_heads)

    # Create some random input tensors
    query = torch.randn(batch_size, 1, hidden_size)
    keys = torch.randn(batch_size, seq_len, 2*hidden_size)

    # Test the forward method
    output = multi_dot(query, keys)
    assert output.shape == (batch_size, 1,hidden_size)

test_multi_dot()
#%%import torch
from model import EncoderND

def test_encoder_nd():
    vocab_size = 100
    embedding_size = 32
    hidden_size = 64
    dropout_ratio = 0.5
    num_layers = 1
    batch_size = 16
    seq_len = 10

    encoder = EncoderND(vocab_size, embedding_size, hidden_size, dropout_ratio, num_layers)

    # Create some random input tensors
    input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    # Test the forward method
    output = encoder(input_tensor)
    assert output.shape == (num_layers, batch_size, seq_len, 2*hidden_size)

test_encoder_nd()
#%%import torch
from model import DecoderND

def test_decoder_nd():
    vocab_size = 100 # V
    embedding_size = 32 # E
    hidden_size = 64 # H
    dropout_ratio = 0.5 
    num_layers = 2
    batch_size = 16 # B
    seq_len = 10 # T

    decoder = DecoderND(vocab_size, embedding_size, hidden_size, dropout_ratio, num_layers)

    # Create some random input tensors
    input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len)) # B,T
    all_hidden_encoder = torch.randn(num_layers, batch_size, seq_len, 2*hidden_size) # B,Tx,2H

    # Test the forward method
    output = decoder(input_tensor, all_hidden_encoder)
    assert output.shape == (batch_size, seq_len, vocab_size)

test_decoder_nd()
# %%
