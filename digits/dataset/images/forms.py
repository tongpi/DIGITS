# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from .job import ImageDatasetJob
from digits import utils
from flask_babel import lazy_gettext as lgt


class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encoding = utils.forms.SelectField(
        lgt('Image Encoding'),
        default='png',
        choices=[
            ('none', lgt('None')),
            ('png', lgt('PNG (lossless)')),
            ('jpg', lgt('JPEG (lossy, 90% quality)')),
        ],
        tooltip=lgt('Using either of these compression formats can save disk space, '
                  'but can also require marginally more time for training.'),
    )

    # Image resize

    resize_channels = utils.forms.SelectField(
        lgt(u'Image Type'),
        default='3',
        choices=[('1', lgt('Grayscale')), ('3', lgt('Color'))],
        tooltip=lgt("Color is 3-channel RGB. Grayscale is single channel monochrome.")
    )
    resize_width = wtforms.IntegerField(
        lgt(u'Resize Width'),
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_height = wtforms.IntegerField(
        lgt(u'Resize Height'),
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_mode = utils.forms.SelectField(
        lgt(u'Resize Transformation'),
        default='squash',
        choices=ImageDatasetJob.resize_mode_choices(),
        tooltip=lgt("Options for dealing with aspect ratio changes during resize. See examples below.")
    )
