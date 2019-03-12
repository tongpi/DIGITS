# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from wtforms import validators

from ..forms import ModelForm
from digits import utils
from flask_babel import lazy_gettext as lgt

class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = utils.forms.IntegerField(
        lgt('Crop Size'),
        validators=[
            validators.NumberRange(min=1),
            validators.Optional()
        ],
        tooltip=lgt("If specified, during training a random square crop will be "
                  "taken from the input image before using as input for the network.")
    )

    use_mean = utils.forms.SelectField(
        lgt('Subtract Mean'),
        choices=[
            ('none', lgt('None')),
            ('image', lgt('Image')),
            ('pixel', lgt('Pixel')),
        ],
        default='image',
        tooltip=lgt("Subtract the mean file or mean pixel for this dataset from each image.")
    )

    aug_flip = utils.forms.SelectField(
        lgt('Flipping'),
        choices=[
            ('none', lgt('None')),
            ('fliplr', lgt('Horizontal')),
            ('flipud', lgt('Vertical')),
            ('fliplrud', lgt('Horizontal and/or Vertical')),
        ],
        default='none',
        tooltip=lgt("Randomly flips each image during batch preprocessing.")
    )

    aug_quad_rot = utils.forms.SelectField(
        lgt('Quadrilateral Rotation'),
        choices=[
            ('none', lgt('None')),
            ('rot90', lgt('0, 90 or 270 degrees')),
            ('rot180', lgt('0 or 180 degrees')),
            ('rotall', lgt('0, 90, 180 or 270 degrees.')),
        ],
        default='none',
        tooltip=lgt("Randomly rotates (90 degree steps) each image during batch preprocessing.")
    )

    aug_rot = utils.forms.IntegerField(
        lgt('Rotation (+- deg)'),
        default=0,
        validators=[
            validators.NumberRange(min=0, max=180)
        ],
        tooltip=lgt("The uniform-random rotation angle that will be performed during batch preprocessing.")
    )

    aug_scale = utils.forms.FloatField(
        lgt('Rescale (stddev)'),
        default=0,
        validators=[
            validators.NumberRange(min=0, max=1)
        ],
        tooltip=lgt("Retaining image size, the image is rescaled with a "
                  "+-stddev of this parameter. Suggested value is 0.07.")
    )

    aug_noise = utils.forms.FloatField(
        lgt('Noise (stddev)'),
        default=0,
        validators=[
            validators.NumberRange(min=0, max=1)
        ],
        tooltip=lgt("Adds AWGN (Additive White Gaussian Noise) during batch "
                  "preprocessing, assuming [0 1] pixel-value range. Suggested value is 0.03.")
    )

    aug_contrast = utils.forms.FloatField(
        lgt('Contrast (factor)'),
        default=0,
        validators=[
            validators.NumberRange(min=0, max=5)
        ],
        tooltip=lgt("Per channel, the mean of the channel is computed and then adjusts each component x "
                  "of each pixel to (x - mean) * contrast_factor + mean. The contrast_factor is picked "
                  "form a random uniform distribution to yield a value between [1-contrast_factor, "
                  "1+contrast_factor]. Suggested value is 0.8.")
    )

    aug_whitening = utils.forms.BooleanField(
        lgt('Whitening'),
        default=False,
        validators=[],
        tooltip=lgt("Per-image whitening by subtracting its own mean, and dividing by its own standard deviation.")
    )

    aug_hsv_use = utils.forms.BooleanField(
        lgt('HSV Shifting'),
        default=False,
        tooltip=lgt("Augmentation by normal-distributed random shifts in HSV "
                  "color space, assuming [0 1] pixel-value range."),
    )
    aug_hsv_h = utils.forms.FloatField(
        lgt('Hue'),
        default=0.02,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=lgt("Standard deviation of a shift that will be performed during "
                  "preprocessing, assuming [0 1] pixel-value range.")
    )
    aug_hsv_s = utils.forms.FloatField(
        lgt('Saturation'),
        default=0.04,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=lgt("Standard deviation of a shift that will be performed during "
                  "preprocessing, assuming [0 1] pixel-value range.")
    )
    aug_hsv_v = utils.forms.FloatField(
        lgt('Value'),
        default=0.06,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=lgt("Standard deviation of a shift that will be performed during "
                  "preprocessing, assuming [0 1] pixel-value range.")
    )
