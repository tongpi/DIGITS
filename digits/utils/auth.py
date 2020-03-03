# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask
import functools
import re
import werkzeug.exceptions

from .routing import get_request_arg, request_wants_json
from digits.utils.permission import admin_permission, super_permission
from flask_babel import lazy_gettext as _


def get_username():
    return get_request_arg('username') or \
        flask.session.get('username', None)


def validate_username(username):
    """
    Raises a ValueError if the username is invalid
    """
    if not username:
        raise ValueError(_('username is required'))
    if not re.match('[a-z]', username):
        raise ValueError(_('Must start with a lowercase letter'))
    if not re.match('[a-z0-9\.\-_]+$', username):
        raise ValueError(_('Only lowercase letters, numbers, periods, dashes and underscores allowed'))


def requires_login(f=None, redirect=True):
    """
    Decorator for views that require the user to be logged in

    Keyword arguments:
    f -- the function to decorate
    redirect -- if True, this function may return a redirect
    """
    if f is None:
        # optional arguments are handled strangely
        return functools.partial(requires_login, redirect=redirect)

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        username = get_username()
        if not username:
            # Handle missing username
            if request_wants_json() or not redirect:
                raise werkzeug.exceptions.Unauthorized()
            else:
                return flask.redirect(flask.url_for('digits.views.login', next=flask.request.path))
        try:
            # Validate username
            validate_username(username)
        except ValueError as e:
            raise werkzeug.exceptions.BadRequest('Invalid username - %s' % str(e))
        return f(*args, **kwargs)
    return decorated


def has_permission(job, action, username=None):
    """
    Returns True if username can perform action on job

    Arguments:
    job -- the Job in question
    action -- the action in question

    Keyword arguments:
    username -- the user in question (defaults to current user)
    """
    if super_permission.can() or admin_permission.can():
        return True

    if job.is_read_only():
        return False

    if username is None:
        username = get_username()

    if not username:
        return False
    if not job.username:
        return True
    return username == job.username
