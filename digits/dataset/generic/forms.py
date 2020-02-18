# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.


import wtforms
from wtforms import validators

from ..forms import DatasetForm
from digits import utils
from flask_babel import lazy_gettext as _


class GenericDatasetForm(DatasetForm):
    """
    Defines the form used to create a new GenericDatasetJob
    """
    # Generic dataset options
    dsopts_feature_encoding = utils.forms.SelectField(
        _('Feature Encoding'),
        default='png',
        choices=[('none', _('None')),
                 ('png', _('PNG (lossless)')),
                 ('jpg', _('JPEG (lossy, 90% quality)')),
                 ],
        tooltip=_("Using either of these compression formats can save disk"
                  " space, but can also require marginally more time for "
                  "training.")
    )

    dsopts_label_encoding = utils.forms.SelectField(
        _('Label Encoding'),
        default='none',
        choices=[
            ('none', _('None')),
            ('png', _('PNG (lossless)')),
            ('jpg', _('JPEG (lossy, 90% quality)')),
        ],
        tooltip=_("Using either of these compression formats can save disk"
                  " space, but can also require marginally more time for"
                  " training.")
    )

    dsopts_batch_size = utils.forms.IntegerField(
        _('Encoder batch size'),
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=32,
        tooltip=_("Encode data in batches of specified number of entries")
    )

    dsopts_num_threads = utils.forms.IntegerField(
        _('Number of encoder threads'),
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=4,
        tooltip=_("Use specified number of encoder threads")
    )

    dsopts_backend = wtforms.SelectField(
        _('DB backend'),
        choices=[
            ('lmdb', 'LMDB'),
        ],
        default='lmdb',
    )

    dsopts_force_same_shape = utils.forms.SelectField(
        _('Enforce same shape'),
        choices=[
            (1, _('Yes')),
            (0, _('No')),
        ],
        coerce=int,
        default=1,
        tooltip=_("Check that each entry in the database has the same shape."
                  "Disabling this will also disable mean image computation.")
    )
