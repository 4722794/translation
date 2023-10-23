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
    output,weights = multi_dot(query, keys)
    assert output.shape == (batch_size, 1,2*hidden_size)

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
# # Old macdonald

# def test_evaluate():
#     vocab_size = 30
#     hidden_size = 64
#     num_heads = 8
#     batch_size = 16
#     seq_len = 10

#     model = DecoderND(vocab_size, 32, hidden_size, 0.5, 2)

#     # Create some random input tensors
#     x_t = torch.randint(low=0,high=vocab_size,size=(batch_size,1))
#     all_hidden_encoder = torch.randn(2, batch_size, seq_len, 2*hidden_size)
#     s_prev = torch.randn(batch_size,1,hidden_size)

#     # Test the evaluate method
#     output = model.evaluate(x_t, all_hidden_encoder, s_prev)
#     assert output.shape == (batch_size, 1, vocab_size)

# test_evaluate()
# %%
# new macdonald
from model import TranslationDNN

def test_translation_dnn():
    vocab_size_s = 100 #V_s
    vocab_size_t = 200 #V_t
    embedding_size = 32 #E
    hidden_size = 64 # H
    dropout_ratio = 0.5
    num_layers = 2 #n
    batch_size = 16 #B
    seq_len_s = 10 # Tx
    seq_len_t = 12 # Ty

    model = TranslationDNN(vocab_size_s, vocab_size_t, embedding_size, hidden_size, dropout_ratio, dropout_ratio, num_layers,dot=False)

    # Create some random input tensors
    x_s = torch.randint(low=0, high=vocab_size_s, size=(batch_size, seq_len_s))
    x_t = torch.randint(low=0, high=vocab_size_t, size=(batch_size, seq_len_t))

    # Test the forward method
    output = model(x_s, x_t)
    assert output.shape == (batch_size, seq_len_t, vocab_size_t)

    # Test the evaluate method
    output = model.evaluate(x_s)
    assert output.shape == (batch_size, 30)

test_translation_dnn()

# %%
import torch
from model import TranslationDNN

def test_translation_dnn():
    vocab_size_s = 100
    vocab_size_t = 200
    embedding_size = 32
    hidden_size = 64
    dropout_ratio = 0.5
    num_layers = 2
    batch_size = 16
    seq_len_s = 10
    seq_len_t = 12

    model = TranslationDNN(vocab_size_s, vocab_size_t, embedding_size, hidden_size, dropout_ratio, dropout_ratio, num_layers,dot=False)

    # Test the forward method
    x_s = torch.randint(low=0, high=vocab_size_s, size=(batch_size, seq_len_s))
    x_t = torch.randint(low=0, high=vocab_size_t, size=(batch_size, seq_len_t))
    output = model(x_s, x_t)
    assert output.shape == (batch_size, seq_len_t, vocab_size_t)

    # Test the evaluate method
    x_s = torch.randint(low=0, high=vocab_size_s, size=(batch_size, seq_len_s))
    output = model.evaluate(x_s)
    assert output.shape == (batch_size, 30)

test_translation_dnn()
#%%
import torch
from model import TranslationDNN

def test_evaluate():
    vocab_size_s = 100 # Vs,Vt
    embedding_size = 32 # E
    hidden_size = 64 # H
    dropout_ratio = 0.5 # d
    num_layers = 2 # n
    batch_size = 16 # B
    seq_len_s = 10 # T

    model = TranslationDNN(vocab_size_s, vocab_size_s, embedding_size, hidden_size, dropout_ratio, dropout_ratio, num_layers, dot=False)

    # Create some random input tensors
    x_s = torch.randint(low=0, high=vocab_size_s, size=(batch_size, seq_len_s))

    # Test the evaluate method
    output = model.evaluate(x_s)
    assert output.shape == (batch_size, 30)

test_evaluate()
#%%