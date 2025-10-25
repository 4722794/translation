"""
Transformer model for machine translation.
Adapted from transformer-translator-pytorch to work with the existing project infrastructure.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Create positional encoding matrix
        pe_matrix = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                else:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('positional_encoding', pe_matrix)

    def forward(self, x):
        # x: (B, L, d_model)
        seq_len = x.size(1)
        x = x * math.sqrt(x.size(-1))
        x = x + self.positional_encoding[:, :seq_len, :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.inf = 1e9

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections and split into heads
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, L, L)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, 1, L) or (B, 1, L, L)
            attn_scores = attn_scores.masked_fill(mask == 0, -self.inf)

        # Softmax and dropout
        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)

        # Multiply by values
        attn_values = torch.matmul(attn_distribs, v)  # (B, num_heads, L, d_k)

        # Concatenate heads and apply final linear
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_values)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_output)

        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self-attention
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(self_attn_output)

        # Cross-attention with encoder output
        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout2(cross_attn_output)

        # Feed-forward
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout3(ff_output)

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class TransformerTranslation(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoder = PositionalEncoder(d_model, max_len)

        # Encoder and Decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        # Output projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src: (B, L)
        # Create mask where padding tokens (0) are False
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt: (B, L)
        batch_size, tgt_len = tgt.size()

        # Padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        # Causal mask (no-peak mask)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        # Combine both masks
        tgt_mask = tgt_padding_mask & tgt_sub_mask  # (B, 1, L, L)

        return tgt_mask

    def forward(self, src, tgt):
        # src: (B, L_src), tgt: (B, L_tgt)

        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Embed and add positional encoding
        src_emb = self.dropout(self.positional_encoder(self.src_embedding(src)))
        tgt_emb = self.dropout(self.positional_encoder(self.tgt_embedding(tgt)))

        # Encode and decode
        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.output_linear(dec_output)

        return output

    def evaluate(self, src, max_len=30):
        """
        Greedy decoding for inference.
        Compatible with the existing evaluation interface.
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Encode source
        src_mask = self.make_src_mask(src)
        src_emb = self.positional_encoder(self.src_embedding(src))
        enc_output = self.encoder(src_emb, src_mask)

        # Start with BOS token (1)
        tgt_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(tgt_tokens)
            tgt_emb = self.positional_encoder(self.tgt_embedding(tgt_tokens))
            dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

            # Get last token prediction
            output = self.output_linear(dec_output[:, -1:, :])
            next_token = output.argmax(dim=-1)

            # Append to sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

            # Check if all sequences have EOS (2)
            if (tgt_tokens == 2).any(dim=1).all():
                break

        # Pad to max_len if needed
        if tgt_tokens.size(1) < max_len:
            padding = torch.zeros(batch_size, max_len - tgt_tokens.size(1),
                                dtype=torch.long, device=device)
            tgt_tokens = torch.cat([tgt_tokens, padding], dim=1)
        else:
            tgt_tokens = tgt_tokens[:, :max_len]

        # Return in format compatible with existing code (tokens, weights)
        # No attention weights for now, return dummy
        weights = torch.zeros(batch_size, max_len, src.size(1), device=device)

        return tgt_tokens, weights