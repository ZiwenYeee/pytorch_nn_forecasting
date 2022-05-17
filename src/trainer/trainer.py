import logging
import inspect
import os
import tempfile
import time
import uuid
from typing import Any, List, Optional, Union, Callable
import numpy as np
# import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

logging.basicConfig(level = 'INFO', # DEBUG
        format = "%(asctime)s %(levelname)s:%(lineno)d] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S")


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names



class Trainer:
    def __init__(
        self,
        target_col = 'target',
        epochs: int = 10,
        learning_rate: float = 1e-3,
        learning_rate_decay_factor: float = 0.5,
        patience: int = 10,
        clip_gradient: float = 10.0,
        device: Optional[Union[torch.device, str]] = None,
#         device: Optional[Union[torch.device, str]] = None,
        early_stopping_patience: int = 10,
    ) -> None:
        
        self.target_col = target_col
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.clip_gradient = clip_gradient
        self.device = device
        self.patience = patience
        self.early_stopping_patience = early_stopping_patience
    
    def __call__(
        self,
        net,
        loss_function,
        train_iter,
        valid_iter = None,
    ) -> None:  # TODO: we may want to return some training information here        
        is_validation_available = valid_iter is not None
        net.to(self.device)
        optimizer = self.get_opt(net)
        scheduler = self.get_scheduler(optimizer)
        
        self.input_names = get_module_forward_input_names(net)
        temp_dir = tempfile.TemporaryDirectory(prefix="ts-temp-")
        epoch_info = {
                "params_path": "%s/%s.pt" % (temp_dir.name, "best_model"),
                "epoch_no": -1,
                "loss": np.Inf,}
        
        epoch_min_loss = np.inf
        patience = 0
        for epoch_no in range(self.epochs):

            epoch_loss = self.loop(epoch_no, net, loss_function, train_iter, optimizer, scheduler, is_train = True)
            if is_validation_available:
                epoch_loss = self.loop(
                    epoch_no, net, loss_function, valid_iter, optimizer, scheduler, is_train=False
                    )
            if epoch_loss < epoch_info['loss']:
                epoch_info['loss'] = epoch_loss
                epoch_info['epoch_no'] = epoch_no
                torch.save(net.state_dict(), epoch_info["params_path"])
                patience = 0
            else:
                patience += 1
            if patience >= self.early_stopping_patience:
                logging.info("Early Stopping.")
                break
                
                
            if not np.isfinite(epoch_loss):
                raise ValueError(
                    "Encountered invalid loss value! Try reducing the learning rate ")

            
                
                
        logging.info("Load the best model.")
        net.load_state_dict(torch.load(epoch_info["params_path"]))
        best_loss = epoch_info['loss']
        best_epoch = epoch_info['epoch_no']
        logging.info(f"Model training is over. best loss: {best_loss:.4f} in Epoch:{best_epoch}")
        
        
        
    def loop(self, epoch_no, net, loss_function, batch_iter, optimizer, scheduler, is_train = True):
        if is_train:
            net.train()
        else:
            net.train(mode = False)
        epoch_loss = 0
        tic = time.time()
        batch_no = 1
        with tqdm(batch_iter, disable = not is_train) as it:
            for batch_no, data_entry in enumerate(it, start = 1):
                optimizer.zero_grad()
                inputs = [data_entry.get(k, None) for k in self.input_names]
                inputs = [k.to(self.device) if k is not None else None for k in inputs]
                
                
                if is_train:
                    output = net(*inputs)
                else:
                    with torch.no_grad():
                        output = net(*inputs)
                
                if isinstance(output, (list, tuple)):
                    output = output[0]
                else:
                    output = output
                    
                labels = data_entry[f"future_{self.target_col}"].to(self.device)
                loss = loss_function(output, labels, is_train, data_entry)

                if is_train:
                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), 
                                                 self.clip_gradient)
                    optimizer.step()

                epoch_loss += loss.item()
            

            lv = epoch_loss/batch_no
            scheduler.step(lv)
            it.set_postfix(
                    ordered_dict={
                        "epoch": f"{epoch_no + 1}/{self.epochs}",
                        ("" if is_train else "validation_")
                        + "avg_epoch_loss": lv,},refresh=False,)
        toc = time.time()
        
        if is_train:
            logging.info("Epoch[%d] Elapsed time %.3f seconds",
                epoch_no,
                (toc - tic),)
        logging.info("Epoch[%d] Evaluation metric '%s'=%.4f",
                epoch_no,
                ("" if is_train else "validation_") + "epoch_loss",
                lv, )
        return lv
    
        
    def get_opt(self, net, opt_name = 'Adam'):
        optimizer = getattr(torch.optim, opt_name)(net.parameters(), 
                                                   lr=self.learning_rate)
        return optimizer
    
    def get_scheduler(self, optimizer):
        lr_scheduler = ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min',
            factor = self.learning_rate_decay_factor,
            patience = self.patience,
            min_lr = 5e-5,
        )
        return lr_scheduler