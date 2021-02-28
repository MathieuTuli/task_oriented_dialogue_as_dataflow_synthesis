"""
MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Any, List

from onmt.utils.logging import logger

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.optim import SGD, Adam, Adagrad, RMSprop
import torch


def get_optimizer_scheduler(
        optim_method: str,
        lr_scheduler: str,
        init_lr: float,
        net_parameters: Any,
        listed_params: List[Any],
        train_loader_len: int,
        max_epochs: int,
        optimizer_kwargs=dict(),
        scheduler_kwargs=dict()) -> torch.nn.Module:
    optimizer = None
    scheduler = None
    optim_processed_kwargs = {
        k: v for k, v in optimizer_kwargs.items() if v is not None}
    scheduler_processed_kwargs = {
        k: v for k, v in scheduler_kwargs.items() if v is not None}
    if optim_method == 'SGD':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " SGD optimizer in config.yaml::**kwargs")
        optimizer = SGD(
            net_parameters, lr=init_lr,
            # momentum=kwargs['momentum'],
            # weight_decay=kwargs['weight_decay']
            **optim_processed_kwargs)
    elif optim_method == 'NAG':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " NAG optimizer  in config.yaml::**kwargs")
        optimizer = SGD(
            net_parameters, lr=init_lr,
            # momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'],
            nesterov=True,
            **optim_processed_kwargs)
    elif optim_method == 'AdaM':
        optimizer = Adam(net_parameters, lr=init_lr,
                         **optim_processed_kwargs)
    elif optim_method == 'AdaGrad':
        optimizer = Adagrad(net_parameters, lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'RMSProp':
        optimizer = RMSprop(net_parameters, lr=init_lr,
                            **optim_processed_kwargs)
    if lr_scheduler == 'StepLR':
        if 'step_size' not in scheduler_processed_kwargs.keys() or \
                'gamma' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'step_size' and 'gamma' need to be specified for"
                "StepLR scheduler in config.yaml::**kwargs")
        scheduler = StepLR(
            optimizer,
            # step_size=kwargs['step_size'], gamma=kwargs['gamma'],
            **scheduler_processed_kwargs)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        # first_restart_epochs = 25
        # increasing_factor = 1
        if 'T_0' not in scheduler_processed_kwargs.keys() or \
                'T_mult' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'first_restart_epochs' and 'increasing_factor' need to be "
                "specified for CosineAnnealingWarmRestarts scheduler in "
                "config.yaml::**kwargs")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_processed_kwargs['T_0'],
            T_mult=scheduler_processed_kwargs['T_mult'],
            **scheduler_processed_kwargs)
    elif lr_scheduler not in ['None', '']:
        logger.critical(f"Unknown LR scheduler {lr_scheduler}")
    return (optimizer, scheduler)
