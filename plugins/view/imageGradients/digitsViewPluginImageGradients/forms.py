# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from digits import utils
from digits.utils import subclass
from flask_wtf import FlaskForm
import wtforms
from wtforms import validators


@subclass
class ConfigForm(FlaskForm):
    """
    A form used to configure gradient visualization
    """

    arrow_color = wtforms.SelectField(
        'Arrow color',
        choices=[
            ('red', 'Red'),
            ('green', 'Green'),
            ('blue', 'Blue'),
        ],
        default='red',
    )

    arrow_size = utils.forms.IntegerField(
        'Arrow size (%)',
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=80,
        tooltip="Expressed as percentage of input image"
    )
