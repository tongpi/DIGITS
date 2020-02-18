# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask_wtf import FlaskForm
import os
from wtforms import validators

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_if_set
from flask_babel import lazy_gettext as _


@subclass
class DatasetForm(FlaskForm):
    """
    A form used to create an image processing dataset
    """

    def validate_folder_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError(_('Folder does not exist or is not reachable'))
            else:
                return True

    train_image_folder = utils.forms.StringField(
        _('Training image folder'),
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder of images to use for training")
    )

    train_label_folder = utils.forms.StringField(
        _('Training label folder'),
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder of training labels")
    )

    val_image_folder = utils.forms.StringField(
        _('Validation image folder'),
        validators=[
            validate_required_if_set('val_label_folder'),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder of images to use for training")
    )

    val_label_folder = utils.forms.StringField(
        _('Validation label folder'),
        validators=[
            validate_required_if_set('val_image_folder'),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder of validation labels")
    )

    resize_image_width = utils.forms.IntegerField(
        _('Resize Image Width'),
        validators=[
            validate_required_if_set('resize_image_height'),
            validators.NumberRange(min=1),
        ],
        tooltip=_("If specified, images will be resized to that dimension after padding")
    )

    resize_image_height = utils.forms.IntegerField(
        _('Resize Image Height'),
        validators=[
            validate_required_if_set('resize_image_width'),
            validators.NumberRange(min=1),
        ],
        tooltip=_("If specified, images will be resized to that dimension after padding")
    )

    padding_image_width = utils.forms.IntegerField(
        _('Padding Image Width'),
        default=1248,
        validators=[
            validate_required_if_set('padding_image_height'),
            validators.NumberRange(min=1),
        ],
        tooltip=_("If specified, images will be padded to that dimension")
    )

    padding_image_height = utils.forms.IntegerField(
        _('Padding Image Height'),
        default=384,
        validators=[
            validate_required_if_set('padding_image_width'),
            validators.NumberRange(min=1),
        ],
        tooltip=_("If specified, images will be padded to that dimension")
    )

    channel_conversion = utils.forms.SelectField(
        _('Channel conversion'),
        choices=[
            ('RGB', 'RGB'),
            ('L', _('Grayscale')),
            ('none', _('None')),
        ],
        default='RGB',
        tooltip=_("Perform selected channel conversion.")
    )

    val_min_box_size = utils.forms.IntegerField(
        _('Minimum box size (in pixels) for validation set'),
        default='25',
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=0),
        ],
        tooltip=_("Retain only the boxes that are larger than the specified "
                  "value in both dimensions. This only affects objects in "
                  "the validation set. Enter 0 to disable this threshold.")
    )

    custom_classes = utils.forms.StringField(
        _('Custom classes'),
        validators=[
            validators.Optional(),
        ],
        tooltip=_("Enter a comma-separated list of class names. "
                  "Class IDs are assigned sequentially, starting from 0. "
                  "Unmapped class names automatically map to 0. "
                  "Leave this field blank to use default class mappings. "
                  "See object detection extension documentation for more "
                  "information.")
    )
