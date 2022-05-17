from functools import partial
from typing import List, Tuple

import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == 'orthogonal':
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass #t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1<0, f'Initialization {initialization} not found'

            
from typing import List, Optional

import torch
import torch.nn as nn


class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dims: List[int],
    ) -> None:
        super().__init__()

        self.__num_features = len(cardinalities)

        def create_embedding(c: int, d: int) -> nn.Embedding:
            embedding = nn.Embedding(c, d)
            return embedding

        self.__embedders = nn.ModuleList(
            [create_embedding(c, d) for c, d in zip(cardinalities, embedding_dims)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length
            # self.__num_features with shape (N,T) or (N)
            cat_feature_slices = torch.chunk(features, self.__num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(
                    self.__embedders, cat_feature_slices
                )
            ],
            dim=-1,
        )


            
class StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        layers = [nn.Dropout(p=0.5), nn.Linear(in_features=in_features, out_features=out_features), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class IdentityBasis(nn.Module):
    def __init__(self,
                 context_length,
                 prediction_length,
                 interpolation_mode: str = 'linear'):
        super().__init__()
        assert (interpolation_mode in ['linear','nearest']) or ('cubic' in interpolation_mode)
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.interpolation_mode = interpolation_mode
    
    def forward(self, theta):
        
        backcast = theta[..., :self.context_length]
        knots = theta[..., self.context_length:]
        
        
        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, 
                                     size=self.prediction_length,
                                     mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, 
                size=self.prediction_length, 
                mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), 
                                    self.prediction_length)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size], 
                    size=self.prediction_length, mode="bicubic"
                )  # , align_corners=True)
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast

class NHITBlock(nn.Module):
    def __init__(self,
                 context_length: int,
                 prediction_length: int,
                 n_theta: int,
                 pooling_size: int,
                 pooling_mode: str = 'max',
                 interpolation_mode: str = 'linear',
                 num_layers: int = 2,
                 hidden_size: list = [512, 512],
                 dropout_rate: float = 0.0,
                 activation: str = 'ReLU',
                 batch_norm: bool = False,
                 cardinalities: List[int] = None,
                 embedding_dims: List[int] = None,
                 unknown_dynamic_feat_num: int = None,
                 known_dynamic_feat_num: int = None,
                ):
        super().__init__()
        assert (str.lower(pooling_mode) in ['max','average', 'mean']), 'Only support ["max", "average"]'
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        assert len(hidden_size) == num_layers, f"hidden size should match num layers"
        
        
        if cardinalities is None:
            cardinalities = None
            embedding_dims = None
        
        if unknown_dynamic_feat_num is None:
            unknown_dynamic_feat_num = 0
        if known_dynamic_feat_num is None:
            known_dynamic_feat_num = 0
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_theta = n_theta
        self.hidden_size = hidden_size
        
        self.pooling_size = pooling_size
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.batch_norm = batch_norm
        
        self.unknown_dynamic_feat_num = unknown_dynamic_feat_num
        self.known_dynamic_feat_num = known_dynamic_feat_num
    
        self.cardinalities = cardinalities
        self.embedding_dims = embedding_dims
        
        context_pooled_length = int(np.ceil(context_length/pooling_size))
        
        activ = getattr(nn, activation)()

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_size,
                                              stride=self.pooling_size, ceil_mode=True)
        elif pooling_mode in ['mean', 'average']:
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_size,
                                              stride=self.pooling_size, ceil_mode=True)
        
        
        
        start_size = context_pooled_length
        if known_dynamic_feat_num is not None:
            if pooling_mode == 'max':
                self.known_pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_size,
                                              stride=self.pooling_size, ceil_mode=True)
            elif pooling_mode in ['mean', 'average']:
                self.known_pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_size,
                                              stride=self.pooling_size, ceil_mode=True)
            full_length = (context_length+prediction_length)
            full_pooled_length = int(np.ceil(full_length/pooling_size))
            
            start_size += full_length * known_dynamic_feat_num
            # start_size += full_pooled_length * known_dynamic_feat_num
        
        if unknown_dynamic_feat_num is not None:

            if pooling_mode == 'max':
                self.unknown_pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_size,
                                                  stride=self.pooling_size, ceil_mode=True)
            elif pooling_mode in ['mean', 'average']:
                self.unknown_pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_size,
                                                  stride=self.pooling_size, ceil_mode=True)
            
            start_size += context_length * unknown_dynamic_feat_num
            # start_size += context_pooled_length * unknown_dynamic_feat_num
            
        if cardinalities is not None:
            static_feat_hidden = sum(embedding_dims)
            start_size += static_feat_hidden
            
        self.hidden_size = [start_size] + self.hidden_size

        
        hidden_layers = []
        for i in range(num_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i+1]))
            hidden_layers.append(activ)

            if self.batch_norm:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i+1]))

            if self.dropout_rate>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_rate))

        output_layer = [nn.Linear(in_features=self.hidden_size[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        # if (self.static_feat_num > 0) and (self.static_feat_hidden > 0):
        #     self.static_encoder = _StaticFeaturesEncoder(in_features=static_feat_num, 
        #                                                  out_features=static_feat_hidden)
        
        self.layers = nn.Sequential(*layers)
        
        self.basis = IdentityBasis(context_length = context_length,
                                   prediction_length = prediction_length,
                                  interpolation_mode = interpolation_mode)
    
    def forward(self, 
                past_target,
                past_unknown_dynamic_feat = None,
                known_dynamic_feat = None,
                static_feat = None,
               ):
        past_target = past_target.unsqueeze(1)
        # Pooling layer to downsample input
        past_target = self.pooling_layer(past_target)
        past_target = past_target.squeeze(1)
        
        batch_size = len(past_target)
        if known_dynamic_feat is not None:
            # known_dynamic_feat = self.known_pooling_layer(known_dynamic_feat.permute(0,2,1)).permute(0,2,1)
            
            past_target = torch.cat(( past_target, known_dynamic_feat.reshape(batch_size, -1) ), 1)
        if past_unknown_dynamic_feat is not None:
            # past_unknown_dynamic_feat = self.unknown_pooling_layer(past_unknown_dynamic_feat.permute(0,2,1)).permute(0,2,1)
            
            past_target = torch.cat(( past_target, past_unknown_dynamic_feat.reshape(batch_size, -1) ), 1)
        # Static exogenous
        if static_feat is not None:
            static_feat = static_feat.reshape(batch_size, -1)
            #static_feat = self.static_encoder(static_feat)
            past_target = torch.cat((past_target, static_feat), 1)
            
        # Compute local projection weights and projection
        theta = self.layers(past_target)
        backcast, forecast = self.basis(theta)

        return backcast, forecast



class NHITS(nn.Module):
    def __init__(self,
                 context_length: int,
                 prediction_length: int,
                 hidden_size: int = 256,
                 pooling_sizes: list = None,
                 downsample_frequencies: list = None,
                 num_blocks: list = [1, 1, 1],
                 num_layers: int = 2,
                 dropout_rate: float = 0.0,
                 pooling_mode: str = 'max',
                 initialization: str = 'lecun_normal',
                 interpolation_mode: str = 'linear',
                 activation: str = 'ReLU',
                 batch_norm: bool = False,
                 shared_weights: bool = False,
                 naive_level: bool = True,
                 cardinalities: List[int] = None,
                 embedding_dims: List[int] = None,
                 unknown_dynamic_feat_num: int = None,
                 known_dynamic_feat_num: int = None,
                ):
        super().__init__()
        # update hparams
        
        num_layers = [num_layers] * len(num_blocks)
        frame = inspect.currentframe()
        for k, v in frame.f_locals.items():
            if k != 'self':
                setattr(self, k, v)
        
        num_stacks = len(num_blocks)
        if pooling_sizes is None:
            pooling_sizes = np.exp2(np.round(np.linspace(0.49, 
                                                              np.log2(prediction_length / 2), 
                                                              num_stacks)))
            pooling_sizes = [int(x) for x in pooling_sizes[::-1]]
            self.pooling_sizes = pooling_sizes
        if downsample_frequencies is None:
            downsample_frequencies = [min(prediction_length, int(np.power(x, 1.5))) 
                                           for x in pooling_sizes]
            self.downsample_frequencies = downsample_frequencies
        
                
        
        block_list = []
        
        for i in range(len(num_blocks)):
            for block_id in range(num_blocks[i]):
                if (len(block_list) == 0) and (batch_norm):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False
                
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = context_length + max(prediction_length // downsample_frequencies[i], 1)
                    nbeats_block = NHITBlock(
                        context_length = context_length,
                        prediction_length = prediction_length,
                        n_theta = n_theta,
                        cardinalities = cardinalities,
                        embedding_dims = embedding_dims,
                        num_layers = num_layers[i],
                        hidden_size = [hidden_size] * num_layers[i],
                        pooling_size = pooling_sizes[i],
                        pooling_mode = pooling_mode,
                        unknown_dynamic_feat_num = unknown_dynamic_feat_num,
                        known_dynamic_feat_num = known_dynamic_feat_num,
                        batch_norm = batch_normalization_block,
                        dropout_rate = dropout_rate,
                        activation = activation,  
                    )              
                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        
        if cardinalities is not None:
            self.embedder = FeatureEmbedder(cardinalities, embedding_dims)
            
        self.blocks = torch.nn.ModuleList(block_list)
        
    def forward(self, 
                past_target,
                past_is_pad,
                past_unknown_dynamic_feat = None,
                known_dynamic_feat = None,
                static_feat = None,
               ):
        
        # residuals = past_target.flip(dims=(-1,))
        # past_is_pad = 1 - past_is_pad.flip(dims=(-1,))
        residuals = past_target
        past_is_pad = 1 - past_is_pad
        
        
        if past_unknown_dynamic_feat is not None:
            # past_unknown_dynamic_feat = past_unknown_dynamic_feat.flip(dims = (-1,))
            past_unknown_dynamic_feat = past_unknown_dynamic_feat
        if known_dynamic_feat is not None:
            # known_dynamic_feat = known_dynamic_feat.flip(dims = (-1,))
            known_dynamic_feat = known_dynamic_feat
        if static_feat is not None:
            static_feat = self.embedder(static_feat)
            
        if self.naive_level:

            forecast = past_target[:, -1:] # Level with Naive1
            block_forecasts = [forecast.repeat(1, self.prediction_length)]
        else:
            forecast = past_target[:, -1:] # Level with Naive1
            forecast = torch.zeros_like(forecast.repeat(1, self.prediction_length), device=forecast.device)
            block_forecasts = []
            
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(past_target=residuals, 
                                             past_unknown_dynamic_feat=past_unknown_dynamic_feat,
                                             known_dynamic_feat=known_dynamic_feat, 
                                             static_feat=static_feat)
            
            residuals = (residuals - backcast) * past_is_pad
            forecast = forecast + block_forecast        
            

            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            
        block_forecasts = torch.stack(block_forecasts, dim=-1)
        return forecast, block_forecasts