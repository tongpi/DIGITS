# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from digits.utils import subclass
from flask_wtf import FlaskForm


@subclass
class ConfigForm(FlaskForm):
    pass
