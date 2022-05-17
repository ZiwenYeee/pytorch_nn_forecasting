from functools import partial
from typing import List, Tuple

import inspect
import numpy as np
import pandas as pd

import torch

from .base import Series

class FeatureProcessor(object):
    def __init__(self,
                 target_field: str,
                 feature_attribute: List[Tuple],
                 normalization = False,
                 norm_type = 'standard'
                ) -> None:
        assert str.lower(norm_type) in ['standard', 'minmax']; "Only support standard and minmax."
        
        frame = inspect.currentframe()
        for k, v in frame.f_locals.items():
            if k != 'self':
                setattr(self, k, v)
                
        self.PreDefineField()
        
    def PreDefineField(self):
        self.target = 'target'
        self.static_feat = 'static_feat'
        self.known_dynamic_feat = 'known_dynamic_feat'
        self.unknown_dynamic_feat = 'unknown_dynamic_feat'
        self.time_feat = 'time_feat'
        
    
    def __call__(self, data):
        if isinstance(data, dict):
            for name, dtype, status in self.feature_attribute + [(self.target_field, 'real', 'unknown')]:
                data = self.ProcessField(data, name, dtype)
            data = self.MergeField(data)
        elif isinstance(data, list):
            for i in range(len(data)):
                tmp_d = data[i].copy()
                for name, dtype, status in self.feature_attribute + [(self.target_field, 'real', 'unknown')]:
                    tmp_d = self.ProcessField(tmp_d, name, dtype)
                tmp_d = self.MergeField(tmp_d)
                data[i] = tmp_d
        return data
        
            
    def MergeField(self, data):
        static_list = []
        known_dynamic_list = []
        unknown_dynamic_list = []
        for col, dtype, status in self.feature_attribute:
            if dtype == 'cat':
                static_list.append(col)
            else:
                if status == 'known':
                    known_dynamic_list.append(col)
                else:
                    unknown_dynamic_list.append(col)
        
        if len(static_list) > 0:
            data[self.static_feat] = np.stack([data.get(col, None) 
                                               for col in static_list], axis = -1)
        
        if len(known_dynamic_list) > 0:
            data[self.known_dynamic_feat] = np.stack([data.get(col, None)
                                                      for col in known_dynamic_list], axis = -1).squeeze()
        if len(unknown_dynamic_list) > 0:
            data[self.unknown_dynamic_feat] = np.stack([data.get(col, None) 
                                                        for col in unknown_dynamic_list], axis = -1).squeeze()
        
        data[self.target] = data[self.target_field].astype(np.float32)
        
        for col in static_list + known_dynamic_list + unknown_dynamic_list + [self.target_field]:
            data.pop(col)

        return data
        
    def ProcessField(self, data, name, is_cat = False):
        dtype = np.int32 if is_cat == 'cat' else np.float32
        value = data.get(name, None)
        if value is not None:
            value = np.asarray(value, dtype=dtype)
            data[name] = value
        return data


def multivariate_transformation(df, 
                             id_col: str,
                             time_col: str,
                             use_cols: list,
                            ):
    df_dict = {}
    for col in use_cols:
        df_values = df.pivot(index=id_col, columns=time_col, values=col)
        df_dict[col] = df_values.values
    
    item_id = df.groupby(id_col).groups.keys()
    df_dict['item_id'] = list(item_id)
    
    df_dict = Series(df_dict)
    return df_dict    
    

def univariate_transformation(df, 
                             id_col: str,
                             time_col: str,
                             use_cols: list,
                             is_train = True,
                             prediction_length = None,
                              
                            ):
    df_dict = {}
    start = None
    for col in use_cols:
        
        df_values = df.pivot(index=id_col, columns=time_col, values=col)
        past_piece = df_values.values
        
        if start is None:
            start = df_values.columns[0]
        if not is_train:
            assert prediction_length is not None, "prediction_length must be provided for test evaluation."
            pad_block = np.ones( (past_piece.shape[0],prediction_length),
                                    dtype=past_piece.dtype) * np.nan
            past_piece = np.concatenate(
                        [past_piece, pad_block], axis=1
                    )
            
            
        df_dict[col] = past_piece
    
    item_id = df.groupby(id_col).groups.keys()
    df_dict['item_id'] = list(item_id)
    df_dict['start'] = len(item_id) * [start]
    
    Field, Value = [], []
    for key, value in df_dict.items():
        if value is not None:
            Field.append(key)
            Value.append(value)

    ds = [Series({key:val for key, val in zip(Field, element)})
                  for element in zip(*Value)]
    return ds