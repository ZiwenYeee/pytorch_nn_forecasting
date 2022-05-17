from typing import (Any, Optional, Iterable, Dict, List, Callable, Union,
                    Tuple, NamedTuple)

import copy
import weakref
import warnings
from collections import namedtuple
from collections.abc import Sequence, Mapping, MutableMapping
#modified from https://github.com/pyg-team/pytorch_geometric/

import numpy as np
import pandas as pd

class Series(MutableMapping):
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self._mapping = {}
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if key == '_parent':
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == '_':
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str):
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any):
        if value is None and key in self:
            del self[key]
        elif value is not None:
            self._mapping[key] = value
    def __delitem__(self, key: str):
        if key in self._mapping:
            del self._mapping[key]

    def __iter__(self) -> Iterable:
        return iter(self._mapping)

    def copy(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def deepcopy(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        out = self.__dict__

        _parent = out.get('_parent', None)
        if _parent is not None:
            out['_parent'] = _parent()

        return out

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

        _parent = self.__dict__.get('_parent', None)
        if _parent is not None:
            self.__dict__['_parent'] = weakref.ref(_parent)
            


    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._mapping.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._mapping.items()]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._mapping.items()]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))

def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):   
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, Series):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'
