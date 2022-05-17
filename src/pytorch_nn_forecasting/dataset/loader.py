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
)

def TrainDataLoader(
    dataset, 
    batch_size, 
    num_batches_per_epoch, 
    pin_memory = False,
    num_workers = 4,
    ):
    total_samples = None if num_batches_per_epoch is None else num_batches_per_epoch * batch_size
    if total_samples is not None:
        sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset,
                                             replacement=True,
                                             num_samples= total_samples)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset)    
    loader = DataLoader(dataset,
                        # collate_fn=batchify,
                        # pin_memory=pin_memory,
                        batch_size=batch_size,
                        sampler = sampler,
                        num_workers=4,
                        )
    return loader


def InferenceDataLoader(
    dataset,
    batch_size,
    num_batches_per_epoch = None, 
    pin_memory = False,
    num_workers = 4,
    ):
    loader = DataLoader(dataset,
                        # collate_fn=batchify,
                        pin_memory=pin_memory,
                        batch_size = batch_size,
                        num_workers=num_workers,
                        shuffle = False,
                        )
    return loader