# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim
import numpy as np

# step 2: define encoder
class EncoderND(nn.Module):
    def __init__(self, V, E, H,dropout_ratio,n):
        super().__init__()
        self.embedding = nn.Embedding(V, E, max_norm=1, scale_grad_by_freq=True)
        self.base_gru = nn.GRU(E, H, batch_first=True, bidirectional=True)
        self.layer_grus = nn.ModuleList([nn.GRU(2*H, H, batch_first=True, bidirectional=True) for _ in range(n-1)])
        self.dropout = nn.Dropout(dropout_ratio) # later improve this by letting this be a heuristic parameter

        # registers
        self.register_buffer("hidden_encoder", torch.zeros(n,1, 1, 2*H))


    def forward(self, x):
        B, T = x.shape
        mask_lens = (x != 0).sum(1).to(torch.device("cpu"))
        emb = self.embedding(x)
        x_pack = pack_padded_sequence(
            emb, mask_lens, batch_first=True, enforce_sorted=False
        )
        all_nh = self.hidden_encoder.repeat(1,B,T,1)
        all_h_packed, _ = self.base_gru(x_pack)
        all_h_unpacked, _ = pad_packed_sequence(all_h_packed, batch_first=True)
        all_nh[0,:,:,:] = self.dropout(all_h_unpacked)
        for n,gru in enumerate(self.layer_grus,1):
            all_h_packed,_ = gru(all_h_packed)
            all_h_unpacked,_ = pad_packed_sequence(all_h_packed, batch_first=True)
            all_nh[n,:,:,:] = self.dropout(all_h_unpacked)
        return all_nh
        # return all hidden states

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)

# step 2: define encoder
class Encoder(nn.Module):
    def __init__(self, V, E, H,dropout_ratio):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(V, E, max_norm=1, scale_grad_by_freq=True)
        self.gru = nn.GRU(E, H, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_ratio) # later improve this by letting this be a heuristic parameter
        self.weight_init()

    def weight_init(self):
        # use a kaiming init
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        B, T = x.shape
        mask_lens = (x != 0).sum(1).to(torch.device("cpu"))
        emb = self.embedding(x)
        x_pack = pack_padded_sequence(
            emb, mask_lens, batch_first=True, enforce_sorted=False
        )
        all_h_packed, _ = self.gru(x_pack)
        all_h, _ = pad_packed_sequence(all_h_packed, batch_first=True)
        all_h = self.dropout(all_h)
        return all_h
        # return all hidden states

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)


# enc = Encoder(8,4,2)
# x = [[4,2,2],[3,5]]
# out = enc(x)
# %%
# create AttentionGRU


class AddAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(2 * hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys):
        """
        keys dim: B,Tx,H
        query dim: B,1,H
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(-1)
        scores = scores.unsqueeze(1)
        # fix the weights
        enc_mask = torch.all((keys == 0), dim=-1).unsqueeze(1)
        scores.masked_fill(enc_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)  # B,1,Tx
        context = weights @ keys  # B,1,H

        return context, weights
    
class DotAttention(nn.Module):

    def __init__(self,hidden_size,head_size):
        super().__init__()
        self.Wq = nn.Linear(hidden_size,head_size)
        self.Wk = nn.Linear(2*hidden_size,head_size)
        self.Wv = nn.Linear(2*hidden_size,2*head_size)
        
        self.head_size = head_size
    def forward(self,query,keys):
        """
        keys dim: B,Tx,H
        query dim: B,1,H
        """
        mod_k = self.Wk(keys) # B,Tx,H
        mod_q = self.Wq(query) # B,1,H
        mod_v = self.Wv(keys) # B,Tx,H
        scores = mod_q @ mod_k.transpose(1,2) # B,1,Tx
        scores = scores / np.sqrt(self.head_size)
        enc_mask = torch.all((keys == 0), dim=-1).unsqueeze(1)
        scores.masked_fill(enc_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        context = weights @ mod_v # B,1,H

        return context, weights

class MultiDot(nn.Module):

    def __init__(self,hidden_size,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([DotAttention(hidden_size,hidden_size//num_heads) for _ in range(num_heads)])

    def forward(self,query,keys):
        """
        keys dim: B,Tx,H
        query dim: B,1,H
        """
        context = torch.cat([head(query,keys)[0] for head in self.heads],dim=-1)
        weights = None # logic to be added later
        return context, weights
#%%
# attn = AddAttention(4)
# q = torch.randn(4,1,4)
# k = torch.randn(4,2,8)
# out = attn(q,k)
# %%

class DecoderND(nn.Module):
    
    def __init__(self,V,E,H,dropout_ratio,n,dot=False):
        super().__init__()

        self.embedding = nn.Embedding(V,E,scale_grad_by_freq=True)
        if dot:
            self.attention = MultiDot(H,8) # hard coding 8 here, but will have to think about it in the future
        else:
            self.attention = AddAttention(H)
        self.base_gru = nn.GRU(E+2*H,H,batch_first=True)
        self.layer_grus = nn.ModuleList([nn.GRU(3*H,H,batch_first=True) for _ in range(n-1)])
        self.out = nn.Linear(H,V)
        self.initialWs = nn.Parameter(torch.randn(n,H,H))
        # dropouts
        self.dropout_emb = nn.Dropout(dropout_ratio)
        self.dropout_att = nn.Dropout(dropout_ratio)
        self.dropout_gru = nn.Dropout(dropout_ratio)
        # number of layers param
        self.num_layers = n
        self.register_buffer("hidden_decoder",torch.zeros(1,1,H))

    def attention_block(self,x,hidden_encoder,layer_num,s_prev=None):
        H = self.base_gru.hidden_size
        B,T,E = x.shape 
        s_prev = s_prev if s_prev is not None else hidden_encoder[:,:1,H:] @ self.initialWs[layer_num] # B,1,H
        hidden_decoder = self.hidden_decoder.repeat(B,T,1)
        for t in range(T):
            x_t = x[:,t,:].unsqueeze(1) # B,1,E for the first case
            c_t, _ = self.attention(s_prev,hidden_encoder)
            c_t = self.dropout_att(c_t)
            x_in = torch.cat((x_t,c_t),dim=-1) # B,T,E+2H for the first one
            if layer_num == 0:
                s_prev,_ = self.base_gru(x_in,s_prev.permute(1,0,2))
            else:
                s_prev,_ = self.layer_grus[layer_num-1](x_in,s_prev.permute(1,0,2))
            s_prev = self.dropout_gru(s_prev)
            hidden_decoder[:,t,:] = s_prev.squeeze(1)
        
        return hidden_decoder

    def forward(self,x_t,all_hidden_encoder,s_prev=None):
        H = self.base_gru.hidden_size
        x_emb = self.embedding(x_t)
        x_emb = self.dropout_emb(x_emb)
        B,T,E = x_emb.shape
        skip_h_decoder = self.hidden_decoder.repeat(B,T,1)
        for n in range(self.num_layers):
            hidden_encoder = all_hidden_encoder[n,:,:,:]
            hidden_decoder = self.attention_block(x_emb,hidden_encoder,n,s_prev)
            skip_h_decoder += hidden_decoder
            x_emb = hidden_decoder
            x_emb = self.dropout_emb(x_emb)
        out = self.out(skip_h_decoder)

        return out
        

    def evaluate(self, x_t, all_hidden_encoder, s_prevs):
        with torch.no_grad():
            H = self.base_gru.hidden_size
            x_emb = self.embedding(x_t)
            B, T, E = x_emb.shape
            skip_h_decoder = self.hidden_decoder.repeat(B, 1, 1)
            new_s_prevs = []
            
            for n in range(self.num_layers):
                hidden_encoder = all_hidden_encoder[n, :, :, :]
                s_prev = s_prevs[n]
                hidden_decoder = self.attention_block(x_emb, hidden_encoder, n, s_prev)
                skip_h_decoder += hidden_decoder
                x_emb = hidden_decoder
                new_s_prevs.append(hidden_decoder)  # The last state is the s_prev
            
            out = self.out(skip_h_decoder)
        
        return out, new_s_prevs

                


class Decoder(nn.Module):
    def __init__(self, V, E, H,dropout_ratio):
        super().__init__()
        self.embedding = nn.Embedding(V, E,scale_grad_by_freq=True)
        self.attention = AddAttention(H)
        self.gru = nn.GRU(E + 2 * H, H, batch_first=True)
        self.out = nn.Linear(H, V)
        self.initialW = nn.Parameter(torch.randn(H, H))
        # dropouts
        self.dropout_emb = nn.Dropout(dropout_ratio)
        self.dropout_att = nn.Dropout(dropout_ratio)
        self.dropout_gru = nn.Dropout(dropout_ratio)
        self.weight_init()
        nn.init.normal_(self.out.weight, 0, 0.01)
        nn.init.zeros_(self.out.bias)

        # buffers
        self.register_buffer("hidden_decoder", torch.zeros(1, 1, H))

    def weight_init(self):
    # use a kaiming init
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x, hidden_encoder):
        H = self.gru.hidden_size
        x_emb = self.embedding(x)
        x_emb = self.dropout_emb(x_emb)
        B, T, E = x_emb.shape
        s_prev = hidden_encoder[:, :1, H:] @ self.initialW  # shape B,1,H
        hidden_decoder = self.hidden_decoder.repeat(B, T, 1)
        for t in range(T):
            x_t = x_emb[:, t, :].unsqueeze(1)
            c_t, _ = self.attention(s_prev, hidden_encoder)
            c_t = self.dropout_att(c_t)
            x_in = torch.cat((x_t, c_t), dim=-1)
            s_prev, _ = self.gru(x_in,s_prev.permute(1,0,2)) # had to permute because s_prev is dims B,1,H but but the input expects to be 1,B,H
            s_prev = self.dropout_gru(s_prev) # questionable dropout, to be reconsidered
            hidden_decoder[:, t, :] = s_prev.squeeze(1)

        out = self.out(hidden_decoder)


        return out

    def evaluate(self, x, hidden_encoder, s_prev):
        with torch.no_grad():
            x_emb = self.embedding(x)
            c_t, weights = self.attention(s_prev, hidden_encoder)
            x_in = torch.cat((x_emb, c_t), dim=-1)
            s_prev, _ = self.gru(x_in,s_prev.permute(1,0,2))
            out = self.out(s_prev)

        return out, s_prev, weights

class TranslationNN(nn.Module):
    def __init__(self, V_s, V_t, E, H,drop_e,drop_d):
        super(TranslationNN, self).__init__()
        self.encoder = Encoder(V_s, E, H,drop_e)
        self.decoder = Decoder(V_t, E, H,drop_d)

        # register buffers

        self.register_buffer("x_init", torch.ones(1, 1).long())

    def forward(self, x_s, x_t):
        all_h_enc = self.encoder(x_s)
        out = self.decoder(x_t, all_h_enc)
        return out

    def evaluate(self, x_s, EOS_token=2,MAXLEN=30):
        with torch.no_grad():
            hidden_encoder = self.encoder.evaluate(x_s)
            B, T, H = hidden_encoder.shape
            H = H // 2  # using bidir
            x_t = self.x_init.repeat(B, 1)
            s_prev = hidden_encoder[:, :1, H:] @ self.decoder.initialW
            counter = 0
            outs = torch.zeros(B, MAXLEN)
            weights = torch.zeros(B, MAXLEN, T)
            while (
                not torch.all(torch.any(outs == EOS_token, dim=1), dim=0).item()
            ) and (counter < MAXLEN):
                out, s_prev, weight = self.decoder.evaluate(x_t, hidden_encoder, s_prev)
                probs = F.softmax(out, dim=-1)
                x_t = torch.argmax(probs, axis=-1)
                outs[:, counter] = x_t.squeeze(1)
                weights[:, counter, :] = weight.squeeze(1)
                counter += 1

        return outs, weights
    

class TranslationDNN(nn.Module):

    def __init__(self,V_s,V_t,E,H,drop_e,drop_d,n,dot=False):
        super().__init__()
        self.encoder = EncoderND(V_s,E,H,drop_e,n)
        self.decoder = DecoderND(V_t,E,H,drop_d,n,dot=dot)
        
        # initialize weights
        self.apply(self._init_weights)
        # register buffers

        self.register_buffer("x_init", torch.ones(1, 1).long())


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, x_s, x_t):
        all_h_enc = self.encoder(x_s)
        out = self.decoder(x_t, all_h_enc)
        return out

    def evaluate(self, x_s, EOS_token=2,MAXLEN=30):
        with torch.no_grad():
            all_hidden_encoder = self.encoder.evaluate(x_s)
            B, _, H = all_hidden_encoder[0].shape
            H = H // 2
            x_t = self.x_init.repeat(B, 1)
            
            s_prevs = [None] * self.decoder.num_layers # daring to pass 'Nones' because I know that in the 'attention_block' code I am creating it if it is None
            counter = 0
            outs = torch.zeros(B, MAXLEN).long()
            
            while not torch.all(torch.any(outs == EOS_token, dim=1)) and counter < MAXLEN:
                out, new_s_prevs = self.decoder.evaluate(x_t, all_hidden_encoder, s_prevs)
                probs = F.softmax(out, dim=-1)
                x_t = torch.argmax(probs, axis=-1)
                outs[:, counter] = x_t.squeeze(1)
                counter += 1
                s_prevs = new_s_prevs  # Update s_prev for the next step
            weights = None
        return outs, weights

            
