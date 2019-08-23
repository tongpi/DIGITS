# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# ********************************************************************
# * 为了新增处理hub格式的模型所创建的文件，不属于官方文件，作者: dzh. *
# ********************************************************************
from __future__ import absolute_import
import os
import hashlib
import tarfile
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask
from flask_babel import lazy_gettext as _


@subclass
class HubUploadTask(UploadPretrainedModelTask):

    def __init__(self, **kwargs):
        super(HubUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return _('Upload Pretrained Tensorflow Hub Model')

    @override
    def __setstate__(self, state):
        super(HubUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        # 根据模型文件与模型url进行模型的解压，解压至指定目录
        if self.model_url and self.model_file:
            model_hash = hashlib.sha1(self.model_url.encode("utf8")).hexdigest()
            f = tarfile.open(self.model_file)
            f.extractall(path='/tmp/tfhub/{}'.format(model_hash))

            self.status = Status.DONE
        else:
            exit(-1)

