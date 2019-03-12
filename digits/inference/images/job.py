# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from ..job import InferenceJob
from digits.utils import subclass, override
from flask_babel import lazy_gettext as lgt


@subclass
class ImageInferenceJob(InferenceJob):
    """
    A Job that exercises the forward pass of an image neural network
    """

    @override
    def job_type(self):
        return lgt('Image Inference')
