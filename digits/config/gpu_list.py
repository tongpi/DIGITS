# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import option_list
import digits.device_query

print('*' * 30)
print(digits.device_query.get_devices())
print('*' * 30)
option_list['gpu_list'] = ','.join([str(x) for x in xrange(len(digits.device_query.get_devices()))])
