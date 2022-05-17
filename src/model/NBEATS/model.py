from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"
VALID_LOSS_FUNCTIONS = "sMAPE", "MASE", "MAPE"


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_block_layers: int,
        expansion_coefficient_length: int,
        prediction_length: int,
        context_length: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.width = width
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_length = expansion_coefficient_length
        self.prediction_length = prediction_length
        self.context_length = context_length
#         self.has_backcast = has_backcast

        fc_stack = [nn.Linear(context_length, width), nn.ReLU()]
        for _ in range(num_block_layers - 1):
            fc_stack.append(nn.Linear(width, width))
            fc_stack.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*fc_stack)

        self.theta_b_fc = nn.Linear(width, expansion_coefficient_length, bias=False)
        self.theta_f_fc = nn.Linear(width, expansion_coefficient_length, bias=False)
        
    def forward(self, x):
        return self.fc_stack(x)
    
class NBEATSGenericBlock(NBEATSBlock):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.backcast_fc = nn.Linear(self.expansion_coefficient_length, 
                                     self.context_length)
        self.forecast_fc = nn.Linear(self.expansion_coefficient_length, 
                                     self.prediction_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
    
    
class NBEATS(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_stacks: int = 4,
        widths: List[int] = [512, 512, 512, 512],
        num_blocks: List[int] = [1, 1, 1, 1],
        num_block_layers: List[int] = [4, 4, 4, 4],
        expansion_coefficient_lengths: List[int] = [32,32,32, 32],
        sharing: List[bool] = [False, False, False, False],
        stack_types: List[str] = ['G', 'G', 'G', 'G'],
        **kwargs,
    ) -> None:
        super(NBEATS, self).__init__()

        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length

        self.net_blocks = nn.ModuleList()
        for stack_id in range(num_stacks):
            for block_id in range(num_blocks[stack_id]):
                net_block = NBEATSGenericBlock(
                        width=self.widths[stack_id],
                        expansion_coefficient_length=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        context_length=context_length,
                        prediction_length=prediction_length,
                    )
                self.net_blocks.append(net_block)

    def forward(self, past_target: torch.Tensor):
        if len(self.net_blocks) == 1:
            _, forecast = self.net_blocks[0](past_target)
            return forecast
        else:
            backcast, forecast = self.net_blocks[0](past_target)
            backcast = past_target - backcast
            for i in range(1, len(self.net_blocks) - 1):
                b, f = self.net_blocks[i](backcast)
                backcast = backcast - b
                forecast = forecast + f
            backcast, last_forecast = self.net_blocks[-1](backcast)
            return forecast + last_forecast
