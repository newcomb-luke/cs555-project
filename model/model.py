#==================================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Kayla
# Description: Transformer-based model for trajectory prediction from waypoint and past state data
#==================================================================================================

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer inputs.
    """

    def __init__(self, device, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd indices

        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Adds positional encodings to input embeddings.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Positional-encoded input
        """

        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    """
    Transformer architecture for flight trajectory prediction.
    Consumes encoded waypoints and observed trajectory segments to predict next flight state.
    """

    def __init__(self, device, d_model=32, nhead=4, num_encoder_layers=3, num_decoder_layers=4, dim_feedforward=64, dropout=0.1, max_seq_len=10):
        super(TrajectoryTransformer, self).__init__()
        self.device = device

        self.input_dim = 6  # (lat, lon, alt, speed_x, speed_y, speed_z)
        self.waypoint_dim = 2 # (lat, lon)

        self.encoder_emb = nn.Linear(self.waypoint_dim, d_model)
        self.pos_encoder = PositionalEncoding(device, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder_emb = nn.Linear(self.input_dim, d_model)
        self.pos_decoder = PositionalEncoding(device, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.output_layer = nn.Linear(d_model, 6)
    
    def forward(self, waypoints, trajectory, tgt_padding_mask=None):
        """
        Runs the model forward pass.

        Args:
            waypoints (Tensor): Tensor of shape (batch, seq_len_wp, 2) with lat/lon waypoints
            trajectory (Tensor): Tensor of shape (batch, seq_len_obs, 6) with past flight states
            tgt_padding_mask (Tensor): Optional mask for target padding tokens

        Returns:
            Tensor: Predicted trajectory point sequences (batch, seq_len_obs, 6)
        """

        # Encoder
        enc_emb = self.encoder_emb(waypoints)
        enc_emb = self.pos_encoder(enc_emb)
        encoder_output = self.encoder(enc_emb)

        # Decoder
        dec_emb = self.decoder_emb(trajectory)
        dec_emb = self.pos_decoder(dec_emb)

        seq_len_tgt = dec_emb.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len_tgt, dec_emb.device)

        decoder_output = self.decoder(
            dec_emb, encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        output = self.output_layer(decoder_output)
        
        return output
    
    def _generate_square_subsequent_mask(self, size, device):
        """
        Creates a causal mask for transformer decoder.

        Args:
            size (int): Size of the square mask
            device (torch.device): Device to create the mask on

        Returns:
            Tensor: (size, size) causal mask
        """

        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Fill with -inf for masking
        return mask