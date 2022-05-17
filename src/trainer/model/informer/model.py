import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import TriangularCausalMask, ProbMask
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding


class Informer(nn.Module):
    #@validated()
    def __init__(self, 
                 input_dim: int,
                 prediction_length: int,
                 context_length: int,
                 label_pad_length: int,
                 freq: str,
                 factor: int = 5, 
                 d_model: int = 512, 
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 d_ff: int = 512,
                 dropout: float = 0.0,
                 attn: str = 'prob',
                 embed: str = 'timeF',
                 activation: str = 'gelu',
                 distil: bool = True,
                 mix: bool = True,
    ) -> None:
        super(Informer, self).__init__()
        
        self.input_dim = input_dim
        self.freq = freq
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.distil = distil
        self.mix = mix
        
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.label_pad_length = label_pad_length
        
        self.attn = attn
        
        self.enc_embedding = DataEmbedding(c_in=input_dim, d_model=d_model, freq=freq, embed_type=embed)
        self.dec_embedding = DataEmbedding(c_in=input_dim, d_model=d_model, freq=freq, embed_type=embed)
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor = factor, 
                                        attention_dropout=dropout, 
                                        output_attention=False), 
                                d_model = d_model, n_heads = n_heads, mix=False),
                    d_model = d_model,
                    d_ff = d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    c_in = d_model
                ) for l in range(e_layers-1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor = factor, 
                                        attention_dropout=dropout, output_attention=False), 
                                d_model = d_model, n_heads = n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor = factor, 
                                                 attention_dropout=dropout, output_attention=False), 
                                d_model = d_model, n_heads = n_heads, mix=False),
                    d_model = d_model,
                    d_ff = d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, input_dim, bias=True)
        
        
    def forward(self, 
                past_target: torch.Tensor, 
                past_time_feat: torch.Tensor, 
                future_time_feat: torch.Tensor,
               ) -> torch.Tensor:
        if len(past_target.shape) == 2:
            past_target = past_target.unsqueeze(-1)
        
        future_pad_target = past_target[:,-self.label_pad_length:,:]
        dec_inp = torch.zeros([past_target.shape[0], 
                               self.prediction_length, 
                               past_target.shape[-1]]).to(past_target.device)
        dec_inp = torch.cat([future_pad_target, dec_inp], dim=1)
        
        future_pad_time_feat = past_time_feat[:, -self.label_pad_length:, :]
        future_time_feat = torch.cat([future_pad_time_feat, future_time_feat], dim = 1)

        
        enc_out = self.enc_embedding(past_target, past_time_feat)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)


        dec_out = self.dec_embedding(dec_inp, future_time_feat)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)


        dec_out = dec_out[:, -self.prediction_length:, :].squeeze()
        return dec_out