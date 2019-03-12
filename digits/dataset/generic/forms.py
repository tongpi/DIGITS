# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from digits import utils
from flask_babel import lazy_gettext as lgt


class GenericDatasetForm(DatasetForm):
    """
    Defines the form used to create a new GenericDatasetJob
    """
    # Generic dataset options
    dsopts_feature_encoding = utils.forms.SelectField(
        lgt('Feature Encoding'),
        default='png',
        choices=[('none', lgt('None')),
                 ('png', lgt('PNG (lossless)')),
                 ('jpg', lgt('JPEG (lossy, 90% quality)')),
                 ],
        tooltip=lgt("Using either of these compression formats can save disk"
                  " space, but can also require marginally more time for "
                  "training.")
    )

    dsopts_label_encoding = utils.forms.SelectField(
        lgt('Label Encoding'),
        default='none',
        choices=[
            ('none', lgt('None')),
            ('png', lgt('PNG (lossless)')),
            ('jpg', lgt('JPEG (lossy, 90% quality)')),
        ],
        tooltip=lgt("Using either of these compression formats can save disk"
                  " space, but can also require marginally more time for"
                  " training.")
    )

    dsopts_batch_size = utils.forms.IntegerField(
        lgt('Encoder batch size'),
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=32,
        tooltip=lgt("Encode data in batches of specified number of entries")
    )

    dsopts_num_threads = utils.forms.IntegerField(
        lgt('Number of encoder threads'),
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=4,
        tooltip=lgt("Use specified number of encoder threads")
    )

    dsopts_backend = wtforms.SelectField(
        lgt('DB backend'),
        choices=[
            ('lmdb', 'LMDB'),
        ],
        default='lmdb',
    )

    dsopts_force_same_shape = utils.forms.SelectField(
        lgt('Enforce same shape'),
        choices=[
            (1, lgt('Yes')),
            (0, lgt('No')),
        ],
        coerce=int,
        default=1,
        tooltip=lgt("Check that each entry in the database has the same shape."
                  "Disabling this will also disable mean image computation.")
    )
