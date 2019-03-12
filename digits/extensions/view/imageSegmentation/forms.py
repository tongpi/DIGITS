# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form

from digits import utils
from digits.utils import subclass
from flask_babel import lazy_gettext as lgt


@subclass
class ConfigForm(Form):
    """
    A form used to display the network output as an image
    """
    colormap = utils.forms.SelectField(
        lgt('Colormap'),
        choices=[
            ('dataset', lgt('From dataset')),
            ('paired', lgt('Paired (matplotlib)')),
            ('none', lgt('None (grayscale)')),
        ],
        default='dataset',
        tooltip=lgt('Set color map to use when displaying segmented image')
    )
