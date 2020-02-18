# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff
from flask_wtf import FlaskForm
import wtforms
from wtforms import validators
from flask_babel import lazy_gettext as _


@subclass
class DatasetForm(FlaskForm):
    """
    A form used to create an image gradient dataset
    """

    train_image_count = utils.forms.IntegerField(
        _('Train Image count'),
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=1000,
        tooltip=_("Number of images to create in training set")
    )

    val_image_count = utils.forms.IntegerField(
        _('Validation Image count'),
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        default=250,
        tooltip=_("Number of images to create in validation set")
    )

    test_image_count = utils.forms.IntegerField(
        _('Test Image count'),
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        default=0,
        tooltip=_("Number of images to create in validation set")
    )

    image_width = wtforms.IntegerField(
        _('Image Width'),
        default=50,
        validators=[validators.DataRequired()]
    )

    image_height = wtforms.IntegerField(
        _('Image Height'),
        default=50,
        validators=[validators.DataRequired()]
    )


@subclass
class InferenceForm(FlaskForm):
    """
    A form used to perform inference on a gradient regression model
    """

    gradient_x = utils.forms.FloatField(
        _('Gradient (x)'),
        validators=[
            validate_required_iff(test_image_count=None),
            validators.NumberRange(min=-0.5, max=0.5),
        ],
        tooltip=_("Specify a number between -0.5 and 0.5")
    )

    gradient_y = utils.forms.FloatField(
        _('Gradient (y)'),
        validators=[
            validate_required_iff(test_image_count=None),
            validators.NumberRange(min=-0.5, max=0.5),
        ],
        tooltip=_("Specify a number between -0.5 and 0.5")
    )

    test_image_count = utils.forms.IntegerField(
        _('Test Image count'),
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        tooltip=_("Number of images to create in test set")
    )
