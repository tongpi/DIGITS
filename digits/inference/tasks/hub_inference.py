# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import base64
from collections import OrderedDict
import h5py
import os.path
import tempfile
import re
import sys

import digits
from digits.task import Task
from digits.utils import subclass, override
from flask_babel import lazy_gettext as _


@subclass
class HubInferenceTask(Task):
    """
    A task for inference jobs
    """
    def __init__(self, model, images, epoch, layers, resize, **kwargs):
        """
        Arguments:
        model  -- trained model to perform inference on
        images -- list of images to perform inference on, or path to a database
        epoch  -- model snapshot to use
        layers -- which layers to visualize (by default only the activations of the last layer)
        """
        # memorize parameters
        self.model = model
        self.images = images
        self.epoch = epoch
        self.layers = layers
        self.resize = resize

        self.image_list_path = None
        self.inference_log_file = "inference.log"

        # resources
        self.gpu = None

        # generated data
        self.inference_data_filename = None
        self.inference_inputs = None
        self.inference_outputs = None
        self.inference_layers = []

        self.infer_result = None

        super(HubInferenceTask, self).__init__(**kwargs)

        # hub
        self.labels = os.path.join(self.model.dir(), 'labels.txt')

    @override
    def name(self):
        return _('Infer Model')

    @override
    def __getstate__(self):
        state = super(HubInferenceTask, self).__getstate__()
        if 'inference_log' in state:
            # don't save file handle
            del state['inference_log']
        return state

    @override
    def __setstate__(self, state):
        super(HubInferenceTask, self).__setstate__(state)

    @override
    def before_run(self):
        super(HubInferenceTask, self).before_run()
        # create log file
        self.inference_log = open(self.path(self.inference_log_file), 'a')
        if type(self.images) is list:
            # create a file to pass the list of images to perform inference on
            imglist_handle, self.image_list_path = tempfile.mkstemp(dir=self.job_dir, suffix='.txt')
            for image_path in self.images:
                os.write(imglist_handle, "%s\n" % image_path)
            os.close(imglist_handle)


    @override
    def process_output(self, line):
        self.inference_log.write('%s\n' % line)
        self.inference_log.flush()

        match = re.match(r'Inference Result:(.*)', line)
        if match:
            self.infer_result = match.group(1)
            return True
        return False

    @override
    def after_run(self):
        super(HubInferenceTask, self).after_run()

        # retrieve inference data
        visualizations = []
        outputs = []
        # roses:0.975007;sunflowers:0.0193765;daisy:0.00269356;dandelion:0.00194239;tulips:0.000980855;
        if self.infer_result:
            for item in self.infer_result.split(';'):
                if not item:
                    break
                outputs.append((item.split(':')[0], round(float(item.split(':')[1]), 4) * 100))

            self.inference_outputs = outputs
            with open(self.images[0], 'r') as f:
                self.inference_inputs = base64.b64encode(f.read()).decode()

        self.inference_log.close()

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from inference_task_pool
        cpu_key = 'inference_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                # we reserve the first available GPU, if there are any
                gpu_key = 'gpus'
                if resources[gpu_key]:
                    for resource in resources[gpu_key]:
                        if resource.remaining() >= 1:
                            self.gpu = int(resource.identifier)
                            reserved_resources[gpu_key] = [(resource.identifier, 1)]
                            break
                return reserved_resources
        return None

    def get_graph(self, step):
        if step is None or 'intermediate_graphintermediate_%d.pb' % step not in os.listdir(self.model.dir()):
            return os.path.join(self.model.dir(), 'frozen_model.pb')
        else:
            return os.path.join(self.model.dir(), 'intermediate_graphintermediate_%d.pb' % step)

    @override
    def task_arguments(self, resources, env):

        # 调用label_image.py进行图片预测
        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'pb', 'label_image.py'),
                '--image=%s' % self.images[0],
                '--graph=%s' % self.get_graph(self.epoch),
                '--labels=%s' % self.labels,
                '--input_layer=Placeholder',
                '--output_layer=final_result'
                ]

        return args

    @classmethod
    def save_result(cls, result):
        cls.INFER_RESULT = result
