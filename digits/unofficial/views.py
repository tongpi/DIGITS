# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import flask
from flask import request, flash, session, redirect, render_template, url_for
from werkzeug import HTTP_STATUS_CODES, secure_filename
from flask_login import login_required
import werkzeug.exceptions
import hashlib

from digits.config import config_value
from digits.webapp import app, socketio, scheduler, db
import digits
from digits import dataset, extensions, model, utils, pretrained_model
from digits.log import logger
from digits.utils.routing import request_wants_json
from digits.models import valid_login, valid_regist, User, verify_pwd
from digits.utils.permission import admin_permission, admin_authority

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/task_manager', methods=['GET'])
@admin_authority
def task_manager():
    running_datasets = get_job_list(dataset.DatasetJob, True)
    completed_datasets = get_job_list(dataset.DatasetJob, False)
    running_models = get_job_list(model.ModelJob, True)
    completed_models = get_job_list(model.ModelJob, False)

    job_data = {
        'datasets': [j.json_dict(True)
                     for j in running_datasets + completed_datasets],
        'models': [j.json_dict(True)
                   for j in running_models + completed_models],
    }
    return render_template('unofficial/task_manager.html', job_data=job_data, scheduler=scheduler, enumerate=enumerate)


def get_job_list(cls, running):
    return sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, cls) and j.status.is_running() == running],
        key=lambda j: j.status_history[0][1],
        reverse=True,
    )


@blueprint.route('/system_manager', methods=['GET'])
@admin_authority
@login_required
def system_manager():
    users = User.get_all_users()
    running_datasets = get_job_list(dataset.DatasetJob, True)
    completed_datasets = get_job_list(dataset.DatasetJob, False)
    running_models = get_job_list(model.ModelJob, True)
    completed_models = get_job_list(model.ModelJob, False)

    job_data = {
        'datasets': [j.json_dict(True)
                     for j in running_datasets + completed_datasets],
        'models': [j.json_dict(True)
                   for j in running_models + completed_models],
    }
    return render_template('unofficial/user_manager.html',
                           users=users,
                           job_data=job_data,
                           scheduler=scheduler,
                           enumerate=enumerate,
                           int=int)


@blueprint.route('/digits.log', methods=['GET'])
def system_log():
    path = config_value('log_file')['filename'].split('/')
    return flask.send_from_directory(directory='/'.join(path[:-1]), filename=path[-1])


@blueprint.route('/index_manager', methods=['GET'])
@login_required
def index_manager():
    completed_datasets = get_job_list(dataset.DatasetJob, False)
    datasets = [j.json_dict(True) for j in completed_datasets]
    dataset_job_ids = [data_set['id'] for data_set in datasets]
    return render_template('unofficial/index_manager.html',
                           ids=dataset_job_ids,
                           scheduler=scheduler,
                           datasets=datasets,
                           hasattr=hasattr)


@blueprint.route('/data_manager', methods=['GET', 'POST'])
@login_required
def data_manager():
    if request.method == 'POST':
        # print(request.form.get('image_folder'))
        # image_file = request.files['image_file']
        image_folder = request.files
        # file_path = request.form.get('file_path')
        # if not file_path:
        #     file_path = '/data/upload_file/'
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # if image_file:
        #     image_file.save(file_path, image_file.filename)
        # print(image_folder)
        # if image_folder:
        # for file in image_folder:
        #     print(file)

        # print(request.form.get('upload_file_path'))
        return redirect(url_for('digits.unofficial.views.data_manager'))
    return render_template('unofficial/data_manager.html')


@blueprint.route('/mark_manager/<index>', methods=['GET'])
@login_required
def mark_manager(index):
    if index == 'image':
        return render_template('unofficial/gds-image.html')
    elif index == 'voice':
        return render_template('unofficial/gds-voice.html')
    elif index == 'video':
        return render_template('unofficial/gds-video.html')
    return render_template('unofficial/mark_manager.html')


@blueprint.route('/visualization_manager', methods=['GET'])
@login_required
def visualization_manager():
    completed_models = get_job_list(model.ModelJob, False)
    models = [j.json_dict(True) for j in completed_models]
    return render_template('unofficial/visualization_manager.html',
                           models=models,
                           scheduler=scheduler,
                           enumerate=enumerate)


@blueprint.route('/add_user', methods=['POST'])
@login_required
def add_user():
    username = request.form.get('username')
    if User.inspect_username(username):
        permissions = request.form.get('permissions')
        password = hashlib.md5(request.form.get('upassword').encode()).hexdigest()
        user = User(username=username, password_hash=password, roles=permissions, status=True)
        db.session.add(user)
        db.session.commit()
    else:
        flash("用户名已存在！请重新输入用户名。")
    return redirect(url_for('digits.unofficial.views.system_manager'))

