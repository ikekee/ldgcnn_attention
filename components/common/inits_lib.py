"""This module contains functions for initializing modules."""
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch

from components.common.configuration import TrainingModelConfiguration
from components.common.configuration import TrainingProcessConfiguration
from components.ldgat_v1_model.ldgat_v1 import LDGATv1
from components.ldgat_v2_model.ldgat_v2 import LDGATv2
from components.ldgcnn_model.ldgcnn_model import LDGCNNSegmentor


def init_optimizer(config: TrainingProcessConfiguration, model_params) -> torch.optim.Optimizer:
    """Initializes an optimizer using provided config and model parameters.

    Args:
        config: TrainingProcessConfiguration object instance.
        model_params: Parameters of an initialized model.

    Returns:
        torch.optim.Optimizer instance according to provided config.
    """
    optimizer_name = config.optimizer
    if optimizer_name == 'adam':
        return torch.optim.Adam(model_params, **config.adam_config)
    else:
        raise NotImplementedError(f"Learning rate scheduler {optimizer_name} is not implemented.")


def init_lr_scheduler(
        config: TrainingProcessConfiguration,
        optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    """Initializes a learning rate scheduler using provided config and optimizer.

    Args:
        config: TrainingProcessConfiguration object instance.
        optimizer: Optimizer instance.

    Returns:
        Instance of torch.optim.lr_scheduler.LRScheduler according to provided config.
    """
    lr_scheduler_name = config.lr_scheduler
    if lr_scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.step)
    else:
        raise NotImplementedError(
            f"Learning rate scheduler {lr_scheduler_name} is not implemented."
        )


def init_model(
        config: TrainingModelConfiguration,
        in_channels: int,
        out_channels: int
) -> Tuple[Union[LDGCNNSegmentor, LDGATv1, LDGATv2], Dict[str, Any]]:
    """Initializes a model according to provided config.

    Args:
        config: TrainingModelConfiguration object instance.
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        Instance of a model according to provided config and its parameters from configration.
    """
    model_name = config.model_to_use
    if model_name == 'ldgcnn':
        model = LDGCNNSegmentor(in_channels=in_channels,
                                out_channels=out_channels,
                                **config.ldgcnn_model_params)
        model_config = config.ldgcnn_model_params
    elif model_name == 'ldgat_v1':
        model = LDGATv1(in_channels=in_channels,
                        out_channels=out_channels,
                        **config.ldgat_v1_model_params)
        model_config = config.ldgat_v1_model_params
    elif model_name == 'ldgat_v2':
        model = LDGATv2(in_channels=in_channels,
                        out_channels=out_channels,
                        **config.ldgat_v2_model_params)
        model_config = config.ldgat_v2_model_params
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
    return model, model_config
