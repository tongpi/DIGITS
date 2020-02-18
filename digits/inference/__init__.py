# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from .images import ImageInferenceJob
from .job import InferenceJob

__all__ = [
    'InferenceJob',
    'ImageInferenceJob',
]
