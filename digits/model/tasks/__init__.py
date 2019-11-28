# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.


from .caffe_train import CaffeTrainTask
from .torch_train import TorchTrainTask
from .train import TrainTask


__all__ = [
    'CaffeTrainTask',
    'TorchTrainTask',
    'TrainTask',
]

from digits.config import config_value  # noqa

if config_value('tensorflow')['enabled']:
    from .tensorflow_train import TensorflowTrainTask  # noqa
    __all__.append('TensorflowTrainTask')
    from .pb_train import PBTrainTask
    __all__.append('PBTrainTask')
    from .hub_train import HubTrainTask
    __all__.append('HubTrainTask')
