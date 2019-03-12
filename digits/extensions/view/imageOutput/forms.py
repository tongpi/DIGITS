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
    channel_order = utils.forms.SelectField(
        lgt('Channel order'),
        choices=[
            ('rgb', 'RGB'),
            ('bgr', 'BGR'),
        ],
        default='rgb',
        tooltip=lgt('Set channel order to BGR for Caffe networks (this field '
                  'is ignored in the case of a grayscale image)')
    )

    data_order = utils.forms.SelectField(
        lgt('Data order'),
        choices=[
            ('chw', 'CHW'),
            ('hwc', 'HWC'),
            ],
        default='chw',
        tooltip=lgt("Set the order of the data. For Caffe and Torch models this is often NCHW"
                  ", for Tensorflow it's NHWC. "
                  "N=Batch Size, W=Width, H=Height, C=Channels")
        )

    pixel_conversion = utils.forms.SelectField(
        lgt('Pixel conversion'),
        choices=[
            ('normalize', lgt('Normalize')),
            ('clip', lgt('Clip')),
        ],
        default='normalize',
        tooltip=lgt('Select method to convert pixel values to the target bit range')
    )

    show_input = utils.forms.SelectField(
        lgt('Show input as image'),
        choices=[
            ('yes', lgt('Yes')),
            ('no', lgt('No')),
        ],
        default='no',
        tooltip=lgt('Show input as image')
    )
