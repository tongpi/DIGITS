# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.


import os

from . import option_list

option_list['url_prefix'] = os.environ.get('DIGITS_URL_PREFIX', '')
