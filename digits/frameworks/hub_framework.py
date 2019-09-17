# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# ********************************************************************
# * 为了新增处理hub格式的模型所创建的文件，不属于官方文件，作者: dzh. *
# ********************************************************************
from __future__ import absolute_import

import os
import re
import subprocess
import tempfile
import sys

from digits.inference.tasks import HubInferenceTask
from .errors import NetworkVisualizationError
from .framework import Framework
import digits
from digits import utils
from digits.model.tasks import HubTrainTask
from digits.utils import subclass, override, constants


@subclass
class HubFramework(Framework):
    """
    Defines required methods to interact with the Tensorflow framework
    """

    # short descriptive name
    NAME = 'TensorflowHub'

    # identifier of framework class
    CLASS = 'tensorflow_hub'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True
    SUPPORTS_PYTHON_LAYERS_FILE = False
    SUPPORTS_TIMELINE_TRACING = True

    SUPPORTED_SOLVER_TYPES = ['SGD', 'ADADELTA', 'ADAGRAD', 'ADAGRADDA', 'MOMENTUM', 'ADAM', 'FTRL', 'RMSPROP']

    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = ['FLIPPING', 'NOISE', 'CONTRAST', 'WHITENING', 'HSV_SHIFTING']

    def __init__(self):
        super(HubFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return HubTrainTask(framework_id=self.framework_id, **kwargs)

    @override
    def create_inference_task(self, **kwargs):
        """
        create inference task
        """
        return HubInferenceTask(**kwargs)




