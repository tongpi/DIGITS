# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from flask.ext.wtf import Form
from wtforms import validators

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff
from flask_babel import lazy_gettext as _


@subclass
class DatasetForm(Form):
    """
    A form used to create an image processing dataset
    """

    def validate_folder_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError(
                    _('Folder does not exist or is not reachable'))
            else:
                return True

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) and not os.path.isdir(field.data):
                raise validators.ValidationError(
                    _('File does not exist or is not reachable'))
            else:
                return True

    feature_folder = utils.forms.StringField(
        _(u'Feature image folder'),
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder full of images.")
    )

    label_folder = utils.forms.StringField(
        _(u'Label image folder'),
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder full of images. For each image in the feature"
                  " image folder there must be one corresponding image in the label"
                  " image folder. The label image must have the same filename except"
                  " for the extension, which may differ. Label images are expected"
                  " to be single-channel images (paletted or grayscale), or RGB"
                  " images, in which case the color/class mappings need to be"
                  " specified through a separate text file.")
    )

    folder_pct_val = utils.forms.IntegerField(
        '% ' + _('for validation'),
        default=10,
        validators=[
            validators.NumberRange(min=0, max=100)
        ],
        tooltip=_("You can choose to set apart a certain percentage of images "
                  "from the training images for the validation set.")
    )

    has_val_folder = utils.forms.BooleanField(_('Separate validation images'),
                                              default=False,
                                              )

    validation_feature_folder = utils.forms.StringField(
        _(u'Validation feature image folder'),
        validators=[
            validate_required_iff(has_val_folder=True),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder full of images.")
    )

    validation_label_folder = utils.forms.StringField(
        _(u'Validation label image folder'),
        validators=[
            validate_required_iff(has_val_folder=True),
            validate_folder_path,
        ],
        tooltip=_("Indicate a folder full of images. For each image in the feature"
                  " image folder there must be one corresponding image in the label"
                  " image folder. The label image must have the same filename except"
                  " for the extension, which may differ. Label images are expected"
                  " to be single-channel images (paletted or grayscale), or RGB"
                  " images, in which case the color/class mappings need to be"
                  " specified through a separate text file.")
    )

    channel_conversion = utils.forms.SelectField(
        _('Channel conversion'),
        choices=[
            ('RGB', 'RGB'),
            ('L', _('Grayscale')),
            ('none', _('None')),
        ],
        default='none',
        tooltip=_("Perform selected channel conversion on feature images. Label"
                  " images are single channel and not affected by this parameter.")
    )

    class_labels_file = utils.forms.StringField(
        _(u'Class labels (optional)'),
        validators=[
            validate_file_path,
        ],
        tooltip=_("The 'i'th line of the file should give the string label "
                  "associated with the '(i-1)'th numeric label. (E.g. the "
                  "string label for the numeric label 0 is supposed to be "
                  "on line 1.)")
    )

    colormap_method = utils.forms.SelectField(
        _('Color map specification'),
        choices=[
            ('label', _('From label image')),
            ('textfile', _('From text file')),
        ],
        default='label',
        tooltip=_("Specify how to map class IDs to colors. Select 'From label "
                  "image' to use palette or grayscale from label images. For "
                  "RGB image labels, select 'From text file' and provide "
                  "color map in separate text file.")
    )

    colormap_text_file = utils.forms.StringField(
        _('Color map file'),
        validators=[
            validate_required_iff(colormap_method="textfile"),
            validate_file_path,
        ],
        tooltip=_("Specify color/class mappings through a text file. "
                  "Each line in the file should contain three space-separated "
                  "integer values, one for each of the Red, Green, Blue "
                  "channels. The 'i'th line of the file should give the color "
                  "associated with the '(i-1)'th class. (E.g. the "
                  "color for class #0 is supposed to be on line 1.)")
    )
