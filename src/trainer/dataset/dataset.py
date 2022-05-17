import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import gc

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    cast,
    Tuple
)

class UnivariateDataset(Dataset):
    def __init__(
        self,
        data_iter: List,
        context_length: int,
        prediction_length: int,
        target_field: str,
        ts_fields_with_attr:List[Tuple] = [],
        valid_method:str = 'holdout',
        ratio: int = 0.9,
        start_idx: int = 0,
        stride: int = 1,
        mode: str = 'train',
        border: List[List] = None,
    ):
        assert mode in ['train', 'valid', 'test', 'inference'];\
        "only support ['train', 'valid', 'test', 'inference]"
        
        if mode == 'inference':
            valid_method = 'holdout'
        
        assert valid_method in ['holdout', 'ratio','train_valid_test'];\
        "Only support ['holdout', 'ratio', train_valid_test]"
        
        if valid_method == 'train_valid_test':
            if border is None:
                raise ValueError("border should be provided.")
            self.border = border
        
        self.data_iter = data_iter
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_field = target_field
        self.ts_fields_with_attr = ts_fields_with_attr
        self.ratio = ratio
        self.mode = mode
        
        if valid_method == 'holdout':
            self.split_idx = self.create_holdout_idx(start_idx, stride)
        elif valid_method == 'ratio':
            self.split_idx = self.create_train_valid_idx(start_idx, stride)
        elif valid_method == 'train_valid_test':
            self.split_idx = self.create_train_valid_test_idx(start_idx, stride)
            
    def _past(self, col_name):
        return f"past_{col_name}"
    def _future(self, col_name):
        return f"future_{col_name}"
    
    def create_holdout_idx(self, start_idx = 0, stride = 1):
        if self.mode == 'train':
            is_train = True
        else:
            is_train = False
            
        split_idx = []
        for seq_num in range(len(self.data_iter)):
            data = self.data_iter[seq_num].copy()
            target = data[self.target_field]
            len_target = target.shape[-1]
            
            if is_train:
                sampling_bounds = (start_idx, len_target - 2 * self.prediction_length, stride)
            else:
                sampling_bounds = (len_target - self.prediction_length, len_target - self.prediction_length + 1, stride)
            
            for idx in range(*sampling_bounds):
                split_idx.append((seq_num, idx))
        return split_idx
    
    def create_train_valid_idx(self, start_idx = 0, stride = 1):
        split_idx = []
        
        if self.mode == 'train':
            is_train = True
        else:
            is_train = False
        
        for seq_num in range(len(self.data_iter)):
            data = self.data_iter[seq_num].copy()
            target = data[self.target_field]
            len_target = target.shape[-1]
            
            end_idx = len_target - self.prediction_length
            
            sampling_bounds = (start_idx, end_idx, stride)
            tmp_idx = np.arange(*sampling_bounds)
            if is_train:
                tmp_idx = tmp_idx[:int(len(tmp_idx) * self.ratio)]
            else:
                tmp_idx = tmp_idx[int(len(tmp_idx) * self.ratio):]
            for idx in tmp_idx:
                split_idx.append((seq_num, idx))
        return split_idx
    def __len__(self):
        return len(self.split_idx)
    
    def base_transform(self, data, idx):
        target = data[self.target_field]
        len_target = target.shape[0]
        slice_cols = self.ts_fields_with_attr + [(self.target_field, 'unknown')]
        
        
        d = data.copy()
        #d = self.process(d)
        
        pad_length = max(self.context_length - idx, 0)
        for ts_field, status in slice_cols:
            if idx >= self.context_length:
                past_piece = d[ts_field][idx - self.context_length:idx, ...]
            else:
                pad_block = np.zeros((pad_length,) + d[ts_field].shape[1:], dtype=d[ts_field].dtype)
                past_piece = d[ts_field][:idx, ...]
                past_piece = np.concatenate(
                        [pad_block, past_piece], axis=0
                    )
            if status == 'unknown':
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][idx:idx+self.prediction_length, ...]
                del d[ts_field]
            else:
                future_piece = d[ts_field][idx:idx+self.prediction_length, ...]
                d[ts_field] = np.concatenate(
                        [past_piece, future_piece], axis=0)
                
        pad_indicator = np.zeros(self.context_length)
        if pad_length > 0:
            pad_indicator[:pad_length] = 1
        d[self._past('is_pad')] = pad_indicator.astype(np.float32)
        
        return d

    def __getitem__(self, idx):
        seq_num, idx = self.split_idx[idx]
        data = self.data_iter[seq_num]
        d = self.base_transform(data, idx)
        d['seq_num'] = np.int32(seq_num)
        d['seq_idx'] = np.int32(idx)
        return d

    
    


# class MultivariateDataset(Dataset):
#     def __init__(
#         self,
#         data_iter: List,
#         context_length: int,
#         prediction_length: int,
#         target_field: str,
#         ts_fields_with_attr:List[Tuple] = [],
#         valid_method = 'holdout',
#         train_ratio = 0.9,
#         start_idx = 0,
#         stride = 1,
#         is_train = True,
#         mode = 'train',
#     ):
#         assert valid_method in ['holdout', 'ratio','train_valid_test'];\
#         "Only support ['holdout', 'ratio', train_valid_test]"
#         #ts_feilds_with_attr must declare future known or not.
        
#         self.data_iter = data_iter
#         self.context_length = context_length
#         self.prediction_length = prediction_length
#         self.target_field = target_field
#         self.ts_fields_with_attr = ts_fields_with_attr
#         self.is_train = is_train
#         self.train_ratio = train_ratio
#         self.mode = mode
        
#         if valid_method == 'holdout':
#             self.split_idx = self.create_holdout_idx(start_idx, stride)
#         elif valid_method == 'ratio':
#             self.split_idx = self.create_train_valid_idx(start_idx, stride)
#         elif valid_method == 'train_valid_test':
#             self.split_idx = self.create_train_valid_test_idx(start_idx, stride)
            
            
#     def __len__(self):
#         return len(self.split_idx)
    
#     def _past(self, col_name):
#         return f"past_{col_name}"
#     def _future(self, col_name):
#         return f"future_{col_name}"
    
#     def _base_transform(self, data, idx):
#         target = data[self.target_field]
#         len_target = target.shape[0]
#         slice_cols = self.ts_fields_with_attr + [(self.target_field, 'unknown')]
        
        
#         d = data.copy()
        
#         pad_length = max(self.context_length - idx, 0)
#         for ts_field, status in slice_cols:
#             if idx >= self.context_length:
#                 past_piece = d[ts_field][idx - self.context_length:idx, ...]
#             else:
#                 pad_block = np.zeros((pad_length,) + d[ts_field].shape[1:], dtype=d[ts_field].dtype)
#                 past_piece = d[ts_field][:idx, ...]
#                 past_piece = np.concatenate(
#                         [pad_block, past_piece], axis=0
#                     )
#             if status == 'unknown':
#                 d[self._past(ts_field)] = past_piece
#                 d[self._future(ts_field)] = d[ts_field][idx:idx+self.prediction_length, ...]
#                 del d[ts_field]
#             else:
#                 future_piece = d[ts_field][idx:idx+self.prediction_length, ...]
#                 d[ts_field] = np.concatenate(
#                         [past_piece, future_piece], axis=0)
                
#         pad_indicator = np.zeros(self.context_length)
#         if pad_length > 0:
#             pad_indicator[:pad_length] = 1
#         d[self._past('is_pad')] = pad_indicator.astype(np.float32)
        
#         return d
    
#     def create_train_valid_idx(self, start_idx = 0, stride = 1):
#         end_idx = self.data_iter[self.target_field].shape[0] - self.prediction_length
#         split_idx = np.arange(start_idx, end_idx, stride)
#         if self.is_train:
#             split_idx = split_idx[:int(end_idx * self.train_ratio)]
#         else:
#             split_idx = split_idx[int(end_idx * self.train_ratio):]
#         return split_idx
    
#     def create_holdout_idx(self, start_idx = 0, stride = 1):
#         len_target = self.data_iter[self.target_field].shape[0]
#         split_idx = []
#         if self.is_train:
#             sampling_bounds = (0, len_target  - 2 * self.prediction_length)
#         else:
#             sampling_bounds = (len_target - self.prediction_length, 
#                                len_target - self.prediction_length + 1)
            
#         for idx in range(*sampling_bounds):
#             split_idx.append(idx)
#         return split_idx
            
#     def __getitem__(self, idx):
#         idx = self.split_idx[idx]
#         d = self._base_transform(self.data_iter, idx)
#         d['pos'] = idx
#         return d
    
#     def create_train_valid_test_idx(self, start_idx = 0, stride = 1):
        
#         target = self.data_iter[self.target_field]
#         len_target = target.shape[0]
#         end_idx = len_target - self.prediction_length

#         sampling_bounds = (start_idx, end_idx, stride)
#         tmp_idx = np.arange(*sampling_bounds)
#         if self.mode == 'train':
#             bound = 6 * 24 * 153
#             tmp_idx = tmp_idx[:bound, ...]
#         elif self.mode == 'valid':
#             bound_l = 6 * 24 * 153
#             bound_u = 6 * 24 * (154 + 16)
#             tmp_idx = tmp_idx[bound_l:bound_u, ...]
#         elif self.mode == 'test':
#             bound_l = 6 * 24 * (154 + 16)
#             tmp_idx = tmp_idx[bound_l:, ...]
                
#         return tmp_idx