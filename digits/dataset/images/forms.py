# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from .job import ImageDatasetJob
from digits import utils
from flask_babel import lazy_gettext as _


class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encoding = utils.forms.SelectField(
        _('Image Encoding'),
        default='png',
        choices=[
            ('none', _('None')),
            ('png', _('PNG (lossless)')),
            ('jpg', _('JPEG (lossy, 90% quality)')),
        ],
        tooltip=_('Using either of these compression formats can save disk space, '
                  'but can also require marginally more time for training.'),
    )

    # Image resize

    resize_channels = utils.forms.SelectField(
        _('Image Type'),
        default='3',
        choices=[('1', _('Grayscale')), ('3', _('Color'))],
        tooltip=_("Color is 3-channel RGB. Grayscale is single channel monochrome.")
    )
    resize_width = wtforms.IntegerField(
        _('Resize Width'),
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_height = wtforms.IntegerField(
        _('Resize Height'),
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_mode = utils.forms.SelectField(
        _('Resize Transformation'),
        default='squash',
        choices=ImageDatasetJob.resize_mode_choices(),
        tooltip=_("Options for dealing with aspect ratio changes during resize. See examples below.")
    )
