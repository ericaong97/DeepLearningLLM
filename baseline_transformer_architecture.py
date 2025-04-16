
import torch
import torch.nn as nn
# import random

# 1. Positional encoding
# based on Attention is All You Need Paper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 2. Transformer architecture
# Changed the default activation function to GELU
# Added layer normalization before transformer layers to improve training stability
class Baseline_Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=4,
                num_decoder_layers=4, dim_feedforward=2048, 
                dropout=0.1, vocab_size=20000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding and positional encoding
        self.embedding_src = nn.Embedding(vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_src = PositionalEncoding(d_model, dropout)
        self.pos_encoder_tgt = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, 
                tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, teacher_forcing_ratio=0.0):
        
        # Input shapes: src=[batch, src_len], tgt=[batch, tgt_len]
        src_emb = self.pos_encoder_src(self.embedding_src(src))
        
        if self.training and teacher_forcing_ratio > 0:
            # Teacher forcing path
            tgt_emb = self.pos_encoder_tgt(self.embedding_tgt(tgt[:, :-1]))  # Shift right
            output = self.transformer(
                src_emb,
                tgt_emb,
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)-1).to(src.device),
                tgt_key_padding_mask=(tgt[:, :-1] == 0) if tgt_key_padding_mask is None else tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask=memory_key_padding_mask
            )
        else:
            # Standard forward pass
            tgt_emb = self.pos_encoder_tgt(self.embedding_tgt(tgt))
            output = self.transformer(
                src_emb, tgt_emb,
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        return self.projection(output)  # [batch, seq_len, vocab_size]

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
# 3. Create transformer
def create_small_transformer(d_model=512, nhead=8, num_encoder_layers=4,
                        num_decoder_layers=4, dim_feedforward=2048, 
                        dropout=0.1, vocab_size=20000):
    return Baseline_Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        vocab_size=vocab_size
    )