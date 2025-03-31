import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        self.device = device
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd indices

        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    def __init__(self, device, d_model=32, nhead=4, num_decoder_layers=4, dim_feedforward=64, dropout=0.1, max_seq_len=10):
        super(TrajectoryTransformer, self).__init__()
        self.input_dim = 10  # (lat, lon, alt, speed_x, speed_y, speed_z, start_lat, start_lon, end_lat, end_lon)
        self.tgt_emb = nn.Linear(self.input_dim, d_model)
        self.pos_encoder_tgt = PositionalEncoding(device, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, 6)
    
    def forward(self, start_end, tgt, tgt_padding_mask=None):
        tgt = torch.cat([tgt, start_end], dim=-1)  # Append start and end points to each step

        tgt_emb = self.tgt_emb(tgt)
        tgt_emb = self.pos_encoder_tgt(tgt_emb)

        seq_len_tgt = tgt_emb.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len_tgt, tgt_emb.device)

        decoder_output = self.transformer_decoder(tgt_emb, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.output_layer(decoder_output)
        
        return output
    
    def _generate_square_subsequent_mask(self, size, device):
        """Creates a causal mask with batch_first=True support."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Fill with -inf for masking
        return mask