# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# ********************************************************************
# * 为了新增处理hub格式的模型所创建的文件，不属于官方文件，作者: dzh. *
# ********************************************************************
from __future__ import absolute_import
import os
import hashlib
import shutil
import tarfile
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask
from flask_babel import lazy_gettext as _


@subclass
class HubUploadTask(UploadPretrainedModelTask):


    def __init__(self, **kwargs):
        self.is_retrain = False
        super(HubUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return _('Upload Pretrained Tensorflow Hub Model')

    def get_model_path(self):
        """
        返回次预训练模型的模型地址
        官方预训练模型：模型URL
        二次训练模型：job_id
        """
        return self.model_url

    def get_model_status(self):
        """
        用于判断是否是官方的预训练模型，如果是官方预训练模型，则为False
        若为二次训练的预训练模型，则为True
        """
        return self.is_retrain

    @override
    def __setstate__(self, state):
        super(HubUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        # 根据模型文件与模型url进行模型的解压，解压至指定目录
        if self.model_url and self.model_file:
            model_hash = hashlib.sha1(self.model_url.encode("utf8")).hexdigest()
            f = tarfile.open(self.model_file)
            # 将模型解压到tfhub目录下，此目录应由环境变量控制，为tensorflow_hub使用的离线模型地址，环境变量名：TFHUB_CACHE_DIR
            # EX: TFHUB_CACHE_DIR="/home/data/tfhub"
            f.extractall(path='/home/data/tfhub/{}'.format(model_hash))

            self.status = Status.DONE
            self.is_retrain = False

            # 用于删除上传的hub模型临时tar.gz文件
            os.remove(self.model_file)

        elif self.before_job_dir and self.model_url:
            if 'https' in self.model_url:
                # 此处model_url为url地址
                model_hash = hashlib.sha1(self.model_url.encode("utf8")).hexdigest()
                shutil.rmtree(self.job_dir)
                shutil.copytree('/home/data/tfhub/{}'.format(model_hash), self.job_dir)
            else:
                # 此处model_url为二次预训练模型的JOB_DIR
                shutil.rmtree(self.job_dir)
                shutil.copytree(self.model_url, self.job_dir)
            self.move_file(os.path.join(self.before_job_dir, 'ckpt.data-00000-of-00001'), 'ckpt.data-00000-of-00001')
            self.move_file(os.path.join(self.before_job_dir, 'ckpt.index'), 'ckpt.index')
            self.move_file(os.path.join(self.before_job_dir, 'ckpt.meta'), 'ckpt.meta')

            self.status = Status.DONE
            self.is_retrain = True
        else:
            exit(1)

