# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
import flask
import tempfile
import tarfile
import zipfile
import json
from flask_login import login_required

import os
import shutil

from digits import utils
from digits.webapp import scheduler
from digits.pretrained_model import PretrainedModelJob
import werkzeug.exceptions
from flask_babel import lazy_gettext as _

blueprint = flask.Blueprint(__name__, __name__)


def get_tempfile(f, suffix):
    temp = tempfile.mkstemp(suffix=suffix)
    f.save(temp[1])
    path = temp[1]
    os.close(temp[0])
    return path


def validate_caffe_files(files):
    """
    Upload a caffemodel
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest(_('Missing weights file'))
    elif files['weights_file'].filename.rsplit('.', 1)[1] != "caffemodel":
        raise werkzeug.exceptions.BadRequest(_('Weights must be a .caffemodel file'))

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest(_('Missing model definition file'))
    elif files['model_def_file'].filename.rsplit('.', 1)[1] != "prototxt":
        raise werkzeug.exceptions.BadRequest(_('Model definition must be .prototxt file'))

    weights_path = get_tempfile(flask.request.files['weights_file'], ".caffemodel")
    model_def_path = get_tempfile(flask.request.files['model_def_file'], ".prototxt")

    return (weights_path, model_def_path)


def validate_tensorflow_files(files):
    """
        Upload a tensorflow model
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing weights file')
    elif files['weights_file'].filename.rsplit('.', 1)[1] != "data-00000-of-00001":
        raise werkzeug.exceptions.BadRequest('Weights must be a .data-00000-of-00001 file')

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing model definition file')
    elif files['model_def_file'].filename != "network.py":
        raise werkzeug.exceptions.BadRequest('Model definition must be network.py file')

    weights_path = get_tempfile(flask.request.files['weights_file'], ".data-00000-of-00001")
    index_path = get_tempfile(flask.request.files['index_file'], ".index")
    meta_path = get_tempfile(flask.request.files['meta_file'], ".meta")
    model_def_path = get_tempfile(flask.request.files['model_def_file'], ".py")

    path_list = [weights_path, index_path, meta_path]

    return (path_list, model_def_path)


def validate_tfpb_files(files):
    """
        Upload a tensorflow pb model
    """
    weights_path = get_tempfile(flask.request.files['weights_file'], ".pb")
    model_def_path = None
    if "model_def_file" in files:
        model_def_path = get_tempfile(flask.request.files['model_def_file'], ".py")

    return (weights_path, model_def_path)


def validate_hub_files(files, form):
    """
        Upload a tensorflow hub model
    """
    model_url = form.get('model_url')
    model_file = get_tempfile(flask.request.files['model_file'], ".tar.gz")

    return (model_url, model_file)


def validate_torch_files(files):
    """
    Upload a torch model
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest(_('Missing weights file'))
    elif files['weights_file'].filename.rsplit('.', 1)[1] != "t7":
        raise werkzeug.exceptions.BadRequest(_('Weights must be a .t7 file'))

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest(_('Missing model definition file'))
    elif files['model_def_file'].filename.rsplit('.', 1)[1] != "lua":
        raise werkzeug.exceptions.BadRequest(_('Model definition must be .lua file'))

    weights_path = get_tempfile(flask.request.files['weights_file'], ".t7")
    model_def_path = get_tempfile(flask.request.files['model_def_file'], ".lua")

    return (weights_path, model_def_path)


def validate_archive_keys(info):
    """
    Validate keys stored in the info.json file
    """
    keys = ["snapshot file", "framework", "name"]
    for key in keys:
        if key not in info:
            return (False, key)

    return (True, 0)


@blueprint.route('/upload_archive', methods=['POST'])
@login_required
def upload_archive():
    """
    Upload archive
    """
    files = flask.request.files
    archive_file = get_tempfile(files["archive"], ".archive")

    if tarfile.is_tarfile(archive_file):
        archive = tarfile.open(archive_file, 'r')
        names = archive.getnames()
    elif zipfile.is_zipfile(archive_file):
        archive = zipfile.ZipFile(archive_file, 'r')
        names = archive.namelist()
    else:
        return flask.jsonify({"status": "Incorrect Archive Type"}), 500

    if "info.json" in names:

        # Create a temp directory to storce archive
        tempdir = tempfile.mkdtemp()
        labels_file = None
        archive.extractall(path=tempdir)

        with open(os.path.join(tempdir, "info.json")) as data_file:
            info = json.load(data_file)

        valid, key = validate_archive_keys(info)

        if valid is False:
            return flask.jsonify({"status": "Missing Key '" + key + "' in info.json"}), 500

        # Get path to files needed to be uploaded in directory
        weights_file = os.path.join(tempdir, info["snapshot file"])

        if "model file" in info:
            model_file = os.path.join(tempdir, info["model file"])
        elif "network file" in info:
            model_file = os.path.join(tempdir, info["network file"])
        else:
            return flask.jsonify({"status": "Missing model definition in info.json"}), 500

        if "labels file" in info:
            labels_file = os.path.join(tempdir, info["labels file"])

        # Upload the Model:
        job = PretrainedModelJob(
            weights_file,
            model_file,
            labels_file,
            info["framework"],
            username=utils.auth.get_username(),
            name=info["name"]
        )

        scheduler.add_job(job)
        job.wait_completion()

        # Delete temp directory
        shutil.rmtree(tempdir, ignore_errors=True)

        return flask.jsonify({"status": "success"}), 200
    else:
        return flask.jsonify({"status": "Missing or Incorrect json file"}), 500


@blueprint.route('/new', methods=['POST'])
@login_required
def new():
    """
    Upload a pretrained model
    """
    labels_path = None
    framework = None
    model_url = None
    model_file = None
    weights_path = ''
    model_def_path = ''

    form = flask.request.form
    files = flask.request.files

    if 'framework' not in form:
        framework = "caffe"
    else:
        framework = form['framework']

    if 'job_name' not in flask.request.form:
        raise werkzeug.exceptions.BadRequest(_('Missing job name'))
    elif str(flask.request.form['job_name']) is '':
        raise werkzeug.exceptions.BadRequest(_('Missing job name'))

    if framework == "caffe":
        weights_path, model_def_path = validate_caffe_files(files)
    elif framework == "torch":
        weights_path, model_def_path = validate_torch_files(files)
    elif framework == "tensorflow":
        weights_path, model_def_path = validate_tensorflow_files(files)
    elif framework == "tensorflow_pb":
        weights_path, model_def_path = validate_tfpb_files(files)
    elif framework == "tensorflow_hub":
        model_url, model_file = validate_hub_files(files, form)
    else:
        exit(1)

    if "labels_file" in files and str(flask.request.files['labels_file'].filename) is not '':
        labels_path = get_tempfile(flask.request.files['labels_file'], ".txt")

    job = PretrainedModelJob(
        weights_path,
        model_def_path,
        labels_path,
        framework,
        form["image_type"],
        form["resize_mode"],
        form["width"],
        form["height"],
        username=utils.auth.get_username(),
        name=flask.request.form['job_name'],
        model_url=model_url,
        model_file=model_file
    )

    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home', tab=3)), 302
