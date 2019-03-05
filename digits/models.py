# -*- coding: utf-8 -*-
from .webapp import db, app
from sqlalchemy import and_
from functools import wraps
from flask import redirect, url_for, session, request
import hashlib


class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(320), unique=True)
    password_hash = db.Column(db.String(128), nullable=True)

    __tablename__ = 'users'

    def __repr__(self):
        return '<User %r>' % self.username


# 登录检验
def valid_login(username, password):
    pwb_hash = hashlib.md5(password.encode()).hexdigest()
    user = User.query.filter(and_(User.username == username, User.password_hash == pwb_hash)).first()
    if user:
        return True
    else:
        return False


# 注册检验
def valid_regist(username):
    user = User.query.filter(User.username == username).first()
    if user:
        return False
    else:
        return True


# 需要登录状态
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if session.get('username'):
            return func(*args, **kwargs)
        else:
            return redirect(url_for('.login', next=request.url))
    return wrapper


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
