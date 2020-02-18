# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.


from flask_wtf import FlaskForm
from wtforms.validators import DataRequired

from digits import utils
from flask_babel import lazy_gettext as _


class DatasetForm(FlaskForm):
    """
    Defines the form used to create a new Dataset
    (abstract class)
    """

    dataset_name = utils.forms.StringField(_('Dataset Name'),
                                           validators=[DataRequired()]
                                           )

    group_name = utils.forms.StringField(_('Group Name'),
                                         tooltip=_("An optional group name for organization on the main page.")
                                         )
