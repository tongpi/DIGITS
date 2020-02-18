# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# ********************************************************************
# * 为了新增处理pb格式的模型所创建的文件，不属于官方文件，作者: dzh. *
# ********************************************************************

import os
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask
from flask_babel import lazy_gettext as _


@subclass
class TfpbUploadTask(UploadPretrainedModelTask):

    def __init__(self, **kwargs):
        super(TfpbUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return _('Upload Pretrained Tensorflow Model')

    @override
    def get_model_def_path(self):
        """
        Get path to model definition
        """
        return os.path.join(self.job_dir, "network.py")

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return os.path.join(self.job_dir, "frozen_model.pb")

    @override
    def __setstate__(self, state):
        super(TfpbUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        self.move_file(self.weights_path, "frozen_model.pb")

        if self.model_def_path is not None:
            self.move_file(self.model_def_path, "network.py")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
