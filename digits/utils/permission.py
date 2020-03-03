"""
@auth: XH
@time: 2020-03-03 9:34
@project: DIGITS
@file: permission.py
@desc:
"""
from flask import flash, redirect, url_for
from functools import wraps
from flask_principal import Permission, RoleNeed


READER = 'READER'
NORMAL = 'NORMAL'
ADMIN = 'ADMIN'
SUPER = 'SUPER'

ROLES = (
    ("READER", "只读用户"),
    ("NORMAL", "普通用户"),
    ("ADMIN", "普通管理员"),
    ("SUPER", "超级管理员")
    )

reader_permission = Permission(RoleNeed(READER))
normal_permission = Permission(RoleNeed(NORMAL))
admin_permission = Permission(RoleNeed(ADMIN))
super_permission = Permission(RoleNeed(SUPER))


def admin_authority(func):
    @wraps(func)
    def decorated_view(*arg, **kwargs):
        if admin_permission.can() or super_permission.can():
            return func(*arg, **kwargs)
        flash("非admin用户，无权限访问！")
        return redirect(url_for('digits.views.home'))
    return decorated_view
