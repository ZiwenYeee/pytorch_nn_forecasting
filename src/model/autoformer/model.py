import torch
import torch.nn as nn
import torch.nn.functional as F
from .embed import DataEmbedding, DataEmbedding_wo_pos, LazyDataEmbedding_wo_pos
from .autocorrelation import AutoCorrelation, AutoCorrelationLayer
from .encoder_decoder import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(
        self,
        context_length,
        prediction_length,
        decoder_output_dim,
        encoder_input_dim = None,
        decoder_input_dim = None,
        label_pad_length = None,
        d_model = 512,
        freq = 'a',
        dropout = 0.05,
        
        embed = 'timeF',
        moving_avg = 25,
        factor = 1,
        activation = 'gelu',
        encoder_layers = 2,
        decoder_layers = 1,
        d_ff = 2048,
        n_heads = 8,
        output_attention = False,
    ):
        super(Autoformer, self).__init__()
        
        if label_pad_length is None:
            label_pad_length = context_length//2
        self.seq_len = context_length
        self.label_len = self.label_pad_length = label_pad_length
        self.pred_len = self.prediction_length = prediction_length
        self.output_attention = output_attention
        
        self.moving_avg = moving_avg
        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)
        
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.c_out = self.decoder_output_dim = decoder_output_dim
        self.d_model = d_model
        self.embed = embed
        self.activation = activation
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        self.e_layers = encoder_layers
        self.d_layers = decoder_layers
        
        self.enc_embedding = LazyDataEmbedding_wo_pos(self.encoder_input_dim, 
                                                  self.d_model, 
                                                  self.embed, 
                                                  self.freq,
                                                  self.dropout)
        self.dec_embedding = LazyDataEmbedding_wo_pos(self.decoder_input_dim, 
                                                  self.d_model, 
                                                  self.embed, 
                                                  self.freq,
                                                  self.dropout)
        
        # self.enc_embedding = DataEmbedding_wo_pos(self.encoder_input_dim, 
        #                                           self.d_model, 
        #                                           self.embed, 
        #                                           self.freq,
        #                                           self.dropout)
        # self.dec_embedding = DataEmbedding_wo_pos(self.decoder_input_dim, 
        #                                           self.d_model, 
        #                                           self.embed, 
        #                                           self.freq,
        #                                           self.dropout)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.factor, attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )
        

    def forward(self, 
                past_target: torch.Tensor,
                known_dynamic_feat: torch.Tensor = None, 
                past_known_dynamic_feat: torch.Tensor = None,
                enc_self_mask: torch.Tensor = None, 
                dec_self_mask: torch.Tensor = None, 
                dec_enc_mask: torch.Tensor = None
               ) -> torch.Tensor:
        # decomp init
        if len(past_target.shape) == 2:
            past_target = past_target.unsqueeze(-1)
        
        if past_known_dynamic_feat is not None:
            past_target = torch.cat([past_target, past_known_dynamic_feat], dim = -1)
        
        past_known_dynamic_feat = known_dynamic_feat[:, :-self.prediction_length, :]
        future_known_dynamic_feat = known_dynamic_feat[:, -self.prediction_length:, :]
        
        
            
        future_pad_target = past_target[:,-self.label_pad_length:,:]
        dec_inp = torch.zeros([past_target.shape[0], 
                               self.prediction_length, 
                               past_target.shape[-1]]).to(past_target.device)
        dec_inp = torch.cat([future_pad_target, dec_inp], dim=1)
        
        future_pad_known_dynamic_feat = past_known_dynamic_feat[:, -self.label_pad_length:, :]
        future_pad_known_dynamic_feat = torch.cat([future_pad_known_dynamic_feat,
                                                   future_known_dynamic_feat], dim = 1)

        x_enc = past_target
        x_mark_enc = past_known_dynamic_feat
        x_dec = dec_inp
        x_mark_dec = future_pad_known_dynamic_feat
        
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :].squeeze(), attns
        else:
            return dec_out[:, -self.pred_len:, :].squeeze()  # [B, L, D]

        