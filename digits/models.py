# -*- coding: utf-8 -*-
from .webapp import db, app, login
from sqlalchemy import and_
from sqlalchemy_utils.types.choice import ChoiceType
from sqlalchemy.dialects import postgresql
from flask import flash, redirect, url_for
from flask_login import UserMixin
import hashlib

from digits.utils.permission import NORMAL, ROLES


class User(db.Model, UserMixin):
    DEFAULT_PERMISSIONS = '0'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(320), unique=True)
    password_hash = db.Column(db.String(128), nullable=True)
    permissions = db.Column(db.String(128), default=DEFAULT_PERMISSIONS)
    status = db.Column(db.Boolean, default=True)

    roles = db.Column(ChoiceType(ROLES), default=NORMAL)

    __tablename__ = 'users'

    def __repr__(self):
        return '<User %r>' % self.username

    @staticmethod
    def get_all_users():
        return User.query.filter().order_by(User.id).all()

    @staticmethod
    def inspect_username(username):
        if User.query.filter(User.username == username).first():
            return False
        else:
            return True

    @staticmethod
    def have_user():
        try:
            u = User.query.filter().all()
            return False if u else True
        except KeyError:
            pass


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


# 登录检验
def valid_login(username, password):
    pwb_hash = hashlib.md5(password.encode()).hexdigest()
    user = User.query.filter(and_(User.username == username, User.password_hash == pwb_hash)).first()
    if user:
        return user
    return None


# 注册检验
def valid_regist(username):
    user = User.query.filter(User.username == username).first()
    if user:
        return False
    else:
        return True


# 两次密码验证
def verify_pwd(pwd, repeat_pwd):
    first_pwd = hashlib.md5(pwd.encode()).hexdigest()
    repeat_pwd = hashlib.md5(repeat_pwd.encode()).hexdigest()
    if first_pwd == repeat_pwd:
        return False
    return True


@app.before_first_request
def create_db():
    db.create_all()
