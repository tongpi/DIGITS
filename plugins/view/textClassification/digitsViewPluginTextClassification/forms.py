# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from digits import utils
from digits.utils import subclass
from flask_wtf import FlaskForm
from wtforms import validators


@subclass
class ConfigForm(FlaskForm):
    """
    A form used to configure text classification visualization
    """

    max_classes = utils.forms.IntegerField(
        'Number of Top classes to show',
        default=5,
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        tooltip='Specify how many classes to show in classification'
    )
