import torch
import torch.nn as nn
import random
from baseline_transformer_architecture import PositionalEncoding

class weight_tying_Transformer(nn.Module):
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
        # Tie weights: projection ↔ target embedding
        self.projection.weight = self.embedding_tgt.weight
        # This ties the weights of the projection to the target embedding layer

        self._init_weights()


    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding_tgt.weight)
        nn.init.xavier_uniform_(self.embedding_src.weight)
        nn.init.constant_(self.projection.bias, 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.projection:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        

    def forward(self, src, tgt, src_key_padding_mask=None, 
                tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, teacher_forcing_ratio=0.0):
        """
        Fixed forward pass with proper attention mask handling
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            [batch_size, tgt_len-1, vocab_size]
        """
        # 1. Source embedding and positional encoding
        src_emb = self.pos_encoder_src(self.embedding_src(src))
        
        # 2. Prepare decoder input (always tgt_len-1)
        decoder_input = tgt[:, :-1]  # Remove last token
        decoder_input_emb = self.pos_encoder_tgt(self.embedding_tgt(decoder_input))
        
        # 3. Generate proper attention mask (CRITICAL FIX)
        seq_len = decoder_input.size(1)  # This should be tgt_len-1 (39)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 4. Handle padding masks (align with decoder input length)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        
        # 5. Transformer forward pass
        output = self.transformer(
            src_emb,
            decoder_input_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,  # Now correctly sized [39,39]
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 6. Project to vocabulary space
        logits = self.projection(output)
        
        # 7. Teacher forcing mixing (training only)
        if self.training and teacher_forcing_ratio > 0 and random.random() < teacher_forcing_ratio:
            return logits
        
        # 8. Free-running generation path (training only)
        if self.training:
            generated = self.generate_sequence(
                src_emb,
                tgt[:, :1],  # Start token
                src_key_padding_mask,
                memory_key_padding_mask,
                max_length=seq_len  # Generate tgt_len-1 tokens
            )
            return generated
        
        return logits  # For validation/inference

    
    def generate_sequence(self, src_emb, start_token, src_key_padding_mask, 
                        memory_key_padding_mask, max_length):
        """Autoregressive sequence generation with proper shape handling"""
        generated = start_token
        outputs = []
        
        for i in range(max_length):
            # 1. Embed and position encode current sequence
            tgt_emb = self.pos_encoder_tgt(self.embedding_tgt(generated))
            
            # 2. Generate mask for current length
            curr_len = generated.size(1)
            tgt_mask = self.generate_square_subsequent_mask(curr_len).to(src_emb.device)
            
            # 3. Forward pass through decoder
            output = self.transformer.decoder(
                tgt_emb,
                src_emb,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,  # No padding during generation
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 4. Get next token prediction
            next_token_logits = self.projection(output[:, -1:])  # [batch, 1, vocab_size]
            outputs.append(next_token_logits)
            
            # 5. Greedy decoding
            next_token = next_token_logits.argmax(-1)
            generated = torch.cat([generated, next_token], dim=1)
        
        # Stack all predictions [batch, max_length, vocab_size]
        return torch.cat(outputs, dim=1)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask with proper device handling"""
        device = next(self.parameters()).device  # Get model device
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask==1, float('-inf'))


    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
# 3. Create transformer
def create_small_weight_tied_transformer(d_model=512, nhead=8, num_encoder_layers=4,
                        num_decoder_layers=4, dim_feedforward=2048, 
                        dropout=0.1, vocab_size=20000):
    return weight_tying_Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        vocab_size=vocab_size
    )