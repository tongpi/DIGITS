# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form
from wtforms.validators import DataRequired

from digits import utils
from flask_babel import lazy_gettext as _


class DatasetForm(Form):
    """
    Defines the form used to create a new Dataset
    (abstract class)
    """

    dataset_name = utils.forms.StringField(_(u'Dataset Name'),
                                           validators=[DataRequired()]
                                           )

    group_name = utils.forms.StringField(_('Group Name'),
                                         tooltip=_("An optional group name for organization on the main page.")
                                         )
