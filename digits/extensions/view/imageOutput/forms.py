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
    channel_order = utils.forms.SelectField(
        _('Channel order'),
        choices=[
            ('rgb', 'RGB'),
            ('bgr', 'BGR'),
        ],
        default='rgb',
        tooltip=_('Set channel order to BGR for Caffe networks (this field '
                  'is ignored in the case of a grayscale image)')
    )

    data_order = utils.forms.SelectField(
        _('Data order'),
        choices=[
            ('chw', 'CHW'),
            ('hwc', 'HWC'),
            ],
        default='chw',
        tooltip=_("Set the order of the data. For Caffe and Torch models this is often NCHW"
                  ", for Tensorflow it's NHWC. "
                  "N=Batch Size, W=Width, H=Height, C=Channels")
        )

    pixel_conversion = utils.forms.SelectField(
        _('Pixel conversion'),
        choices=[
            ('normalize', _('Normalize')),
            ('clip', _('Clip')),
        ],
        default='normalize',
        tooltip=_('Select method to convert pixel values to the target bit range')
    )

    show_input = utils.forms.SelectField(
        _('Show input as image'),
        choices=[
            ('yes', _('Yes')),
            ('no', _('No')),
        ],
        default='no',
        tooltip=_('Show input as image')
    )
