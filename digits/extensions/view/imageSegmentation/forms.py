# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


from flask_wtf import FlaskForm

from digits import utils
from digits.utils import subclass
from flask_babel import lazy_gettext as _


@subclass
class ConfigForm(FlaskForm):
    """
    A form used to display the network output as an image
    """
    colormap = utils.forms.SelectField(
        _('Colormap'),
        choices=[
            ('dataset', _('From dataset')),
            ('paired', _('Paired (matplotlib)')),
            ('none', _('None (grayscale)')),
        ],
        default='dataset',
        tooltip=_('Set color map to use when displaying segmented image')
    )
