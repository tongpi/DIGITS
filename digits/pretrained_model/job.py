# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import hashlib
import os

from digits.job import Job
from digits.utils import subclass, override
from digits.pretrained_model.tasks import CaffeUploadTask, TorchUploadTask, TensorflowUploadTask, TfpbUploadTask, HubUploadTask
from flask_babel import lazy_gettext as _


@subclass
class PretrainedModelJob(Job):
    """
    A Job that uploads a pretrained model
    """

    def __init__(self, weights_path, model_def_path, labels_path=None, framework="caffe",
                 image_type="3", resize_mode="Squash", width=224, height=224, **kwargs):
        super(PretrainedModelJob, self).__init__(persistent=False, **kwargs)

        self.framework = framework
        self.model_url = kwargs.get('model_url', None)
        self.model_path = hashlib.sha1(self.model_url.encode("utf8")).hexdigest() if self.model_url else None
        self.model_file = kwargs.get('model_file', None)
        self.before_job_dir = kwargs.get('before_job_dir', None)

        self.image_info = {
            "image_type": image_type,
            "resize_mode": resize_mode,
            "width": width,
            "height": height
        }

        self.tasks = []

        taskKwargs = {
            "weights_path": weights_path,
            "model_def_path": model_def_path,
            "image_info": self.image_info,
            "labels_path": labels_path,
            "job_dir": self.dir(),
            "model_url": self.model_url,  # hub新增参数，模型地址
            "model_file": self.model_file,  # hub新增参数，模型压缩文件目录
            'before_job_dir': self.before_job_dir  # hub新增参数，已训练好的模型目录
        }

        if self.framework == "caffe":
            self.tasks.append(CaffeUploadTask(**taskKwargs))
        elif self.framework == "torch":
            self.tasks.append(TorchUploadTask(**taskKwargs))
        elif self.framework == "tensorflow":
            self.tasks.append(TensorflowUploadTask(**taskKwargs))
        elif self.framework == "tensorflow_pb":
            self.tasks.append(TfpbUploadTask(**taskKwargs))
        elif self.framework == "tensorflow_hub":
            # 创建hub预训练模型的任务
            self.tasks.append(HubUploadTask(**taskKwargs))
        else:
            raise Exception(_("framework of type %(framework)s is not supported", framework=self.framework))

    def get_weights_path(self):
        return self.tasks[0].get_weights_path()

    def get_model_def_path(self):
        return self.tasks[0].get_model_def_path()

    # dzh: 由创建模型的视图函数调用，获取hub模型的URL
    def get_model_path(self):
        return self.tasks[0].get_model_path()

    def get_model_status(self):
        # 获取模型是否为再训练模型
        return self.tasks[0].get_model_status()

    def get_python_layer_path(self):
        tmp_dir = os.path.dirname(self.tasks[0].get_model_def_path())
        python_layer_file_name = 'digits_python_layers.py'
        if os.path.exists(os.path.join(tmp_dir, python_layer_file_name)):
            return os.path.join(tmp_dir, python_layer_file_name)
        elif os.path.exists(os.path.join(tmp_dir, python_layer_file_name + 'c')):
            return os.path.join(tmp_dir, python_layer_file_name + 'c')
        else:
            return None

    def has_labels_file(self):
        return os.path.isfile(self.tasks[0].get_labels_path())

    @override
    def is_persistent(self):
        return True

    @override
    def job_type(self):
        return _("Pretrained Model")

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name', 'username', 'tasks', 'status_history', 'framework', 'image_info']
        full_state = super(PretrainedModelJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    @override
    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)
