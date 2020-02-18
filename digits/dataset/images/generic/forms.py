# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff
from flask_babel import lazy_gettext as _


class GenericImageDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new GenericImageDatasetJob
    """

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(
        _('Dataset type'),
        choices=[
            ('prebuilt', _('Prebuilt')),
        ],
        default='prebuilt',
    )

    def validate_lmdb_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError(_('Folder does not exist'))

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isfile(field.data):
                raise validators.ValidationError(_('File does not exist'))

    #
    # Method - prebuilt
    #

    prebuilt_train_images = wtforms.StringField(
        'Training Images',
        validators=[
            validate_required_iff(method='prebuilt'),
            validate_lmdb_path,
        ]
    )
    prebuilt_train_labels = wtforms.StringField(
        'Training Labels',
        validators=[
            validate_lmdb_path,
        ]
    )
    prebuilt_val_images = wtforms.StringField(
        'Validation Images',
        validators=[
            validate_lmdb_path,
        ]
    )
    prebuilt_val_labels = wtforms.StringField(
        'Validation Labels',
        validators=[
            validate_lmdb_path,
        ]
    )

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    force_same_shape = utils.forms.SelectField(
        _('Enforce same shape'),
        choices=[
            (1, _('Yes')),
            (0, _('No')),
        ],
        coerce=int,
        default=1,
        tooltip=_('Check that each entry in the database has the same shape (can be time-consuming)')
    )

    prebuilt_mean_file = utils.forms.StringField(
        _('Mean Image'),
        validators=[
            validate_file_path,
        ],
        tooltip=_("Path to a .binaryproto file on the server")
    )
