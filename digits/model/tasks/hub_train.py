# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# ********************************************************************
# * 为了新增处理hub格式的模型所创建的文件，不属于官方文件，作者: dzh. *
# ********************************************************************


import operator
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys

import h5py
import numpy as np

from .train import TrainTask
import digits
from digits import utils
from digits.utils import subclass, override, constants
import tensorflow as tf
from flask_babel import lazy_gettext as _
from functools import reduce

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

# Constants
TENSORFLOW_MODEL_FILE = 'checkpoint'
TENSORFLOW_SNAPSHOT_PREFIX = 'snapshot'
TENSORFLOW_MODEL_PB = 'frozen_model.pb'
TIMELINE_PREFIX = 'timeline'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def subprocess_visible_devices(gpus):
    """
    Calculates CUDA_VISIBLE_DEVICES for a subprocess
    """
    if not isinstance(gpus, list):
        raise ValueError(_('gpus should be a list'))
    gpus = [int(g) for g in gpus]

    old_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if old_cvd is None:
        real_gpus = gpus
    else:
        map_visible_to_real = {}
        for visible, real in enumerate(old_cvd.split(',')):
            map_visible_to_real[visible] = int(real)
        real_gpus = []
        for visible_gpu in gpus:
            real_gpus.append(map_visible_to_real[visible_gpu])
    return ','.join(str(g) for g in real_gpus)


@subclass
class HubTrainTask(TrainTask):
    """
    Trains a tensorflow model
    """

    TENSORFLOW_LOG = 'tensorflow_hub_output.log'


    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """
        super(HubTrainTask, self).__init__(**kwargs)

        # save network description to file
        # with open(os.path.join(self.job_dir, TENSORFLOW_MODEL_FILE), "w") as outfile:
        #     outfile.write(self.network)

        self.pickver_task_tensorflow_train = PICKLE_VERSION

        self.current_epoch = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.solver = None

        self.model_file = TENSORFLOW_MODEL_FILE
        self.train_file = constants.TRAIN_DB
        self.val_file = constants.VAL_DB
        self.model_pb = TENSORFLOW_MODEL_PB
        self.snapshot_prefix = TENSORFLOW_SNAPSHOT_PREFIX
        self.log_file = self.TENSORFLOW_LOG

    def __getstate__(self):
        state = super(HubTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'tensorflow_log' in state:
            del state['tensorflow_log']

        return state

    def __setstate__(self, state):
        super(HubTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None

    # Task overrides
    @override
    def name(self):
        return 'Train Tensorflow Hub Model'

    @override
    def before_run(self):
        super(HubTrainTask, self).before_run()
        self.tensorflow_log = open(self.path(self.TENSORFLOW_LOG), 'a')
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        self.displaying_network = False
        self.temp_unrecognized_output = []
        return True

    @override
    def get_snapshot(self, epoch=-1, download=False, frozen_file=False):
        """
        return snapshot file for specified epoch
        """
        graph_name = None

        if epoch == -1 or not epoch:
            if self.snapshots:
                graph_name = self.snapshots[-1][0]
            else:
                graph_name = 'snapshot.ckpt'
        else:
            for f, e in self.snapshots:
                if e == epoch:
                    graph_name = f
                    break
        if not graph_name:
            raise ValueError(_('Invalid epoch'))

        if download:
            snapshot_files = os.path.join(self.job_dir, graph_name)
        else:
            snapshot_files = os.path.join(self.job_dir, 'snapshot.ckpt')
        return snapshot_files

    @override
    def task_arguments(self, resources, env):
        """
        :tfhub_module: hub中模型的地址（URL）
        :image_dir: 进行再训练时的图像路径
        :output_graph: 训练完成的graph文件路径
        :output_labels: 训练完成的标签文件路径
        :bottleneck_dir: 瓶颈值缓存路径
        :checkpoint_path: ckpt文件保存路径
        :intermediate_store_frequency: 存储步骤数，根据步骤来进行片段存储，默认为0
        :intermediate_output_graphs_dir: 模型中间片段存储路径
        :train_batch_size：批处理大小
        total: 25`args
        """

        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'pb', 'retrain.py'),
                '--tfhub_module=%s' % self.tfhub_module,
                '--image_dir=%s' % self.image_dir,
                '--output_graph=%s' % self.output_graph,
                '--intermediate_output_graphs_dir=%s' % self.intermediate_output_graphs_dir,
                '--output_labels=%s' % self.output_labels,
                '--bottleneck_dir=%s' % self.bottleneck_dir,
                '--checkpoint_path=%s' % os.path.join(self.job_dir, self.checkpoint_path),
                '--summaries_dir=%s' % self.summaries_dir,]

        if self.save_ckpt_path:
            args.append('--save_ckpt_path=%s' % self.save_ckpt_path)

        if self.intermediate_store_frequency:
            args.append('--intermediate_store_frequency=%s' % int(self.intermediate_store_frequency))

        if self.learning_rate:
            args.append('--learning_rate=%f' % self.learning_rate)

        if self.how_many_training_steps:
            args.append('--how_many_training_steps=%d' % self.how_many_training_steps)

        if self.train_batch_size:
            args.append('--train_batch_size=%d' % self.train_batch_size)

        if 'gpus' in resources:
            identifiers = []
            for identifier, value in resources['gpus']:
                identifiers.append(identifier)
            # make all selected GPUs visible to the process.
            # don't make other GPUs visible though since the process will load
            # CUDA libraries and allocate memory on all visible GPUs by
            # default.
            env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices(identifiers)

        # if self.training_steps:
        #     # 训练步骤
        #     args.append('--how_many_training_steps=%s' % self.training_steps)
        #
        #
        # if self.testing_percentage:
        #     # 使用多少图像作为测试集
        #     args.append('--testing_percentage=%d' % self.testing_percentage)
        #
        # if self.validation_percentage:
        #     # 使用多少图像作为验证集
        #     args.append('--validation_percentage=%d' % self.validation_percentage)
        #
        # if self.eval_step_interval:
        #     # 多久评估一次培训结果
        #     args.append('--eval_step_interval=%d' % self.eval_step_interval)
        #
        # if self.train_batch_size:
        #     args.append('--train_batch_size=%d' % self.train_batch_size)
        #
        # if self.test_batch_size:
        #     args.append('--test_batch_size=%d' % self.test_batch_size)
        #
        # if self.validation_batch_size:
        #     args.append('--validation_batch_size=%d' % self.validation_batch_size)
        #
        # if self.print_misclassified_test_images:
        #     # 是否打印出所有错误分类的测试图像的列表
        #     args.append('--print_misclassified_test_images=%s' % self.print_misclassified_test_images)
        #
        # if self.final_tensor_name:
        #     # 再训练图中输出分类图层的名称
        #     args.append('--final_tensor_name=%s' % self.final_tensor_name)
        #
        # if self.flip_left_right:
        #     # 是否水平地随机翻转一半训练图像
        #     args.append('--flip_left_right=%s' % self.flip_left_right)
        #
        # if self.random_crop:
        #     # 随机剪裁
        #     args.append('--random_crop=%d' % self.random_crop)
        # if self.random_scale:
        #     # 随机扩大比例
        #     args.append('--random_scale=%d' % self.random_scale)
        # if self.random_brightness:
        #     # 向上向下
        #     args.append('--random_brightness=%d' % self.random_brightness)
        #
        # if self.saved_model_dir:
        #     # 保存graph输出的位置
        #     args.append('--saved_model_dir=%s' % self.saved_model_dir)

        return args

    def send_update(self):
        self.detect_snapshots()
        self.send_snapshot_update()
        return True

    @override
    def process_output(self, line):
        self.tensorflow_log.write('%s\n' % line)
        self.tensorflow_log.flush()

        # parse tensorflow output
        timestamp, level, step, message = self.preprocess_output_tensorflow(line)

        # return false when unrecognized output is encountered
        if not level:
            # network display in progress
            if self.displaying_network:
                self.temp_unrecognized_output.append(line)
                return True
            return False

        if not message:
            return True

        # network display ends
        if self.displaying_network:
            if message.startswith('Network definition ends'):
                self.temp_unrecognized_output = []
                self.displaying_network = False
            return True

        # 此处为过滤retrain.py创建模型时的输出
        self.train_epochs = self.how_many_training_steps
        pattern_stage_epoch = re.compile(r'(Validation|Train|Cross)\ (.*)')
        for (stage, kvlist) in re.findall(pattern_stage_epoch, message):
            # accuracy = 92.0%
            self.send_progress_update(step)
            pattern_key_val = re.compile(r'([\w]+)\ =\ (\S+)')
            # m = re.match(r'([\w]+)\ =\ (\S+)', 'accuracy = 92.0%')
            # Now iterate through the keys and values on this line dynamically
            for (key, value) in re.findall(pattern_key_val, kvlist):
                if stage == 'Train':
                    # 此处为进度条数据的创建
                    value = float(value.split('%')[0]) / 100
                    self.save_train_output(key, key, value)
                    self.save_train_output('learning_rate', 'learning_rate', self.learning_rate)
                elif stage == 'Validation':
                    value = float(value.split('%')[0]) / 100
                    self.save_val_output(key, key, value)
                elif stage == 'Cross':
                    value = float(value)
                    self.save_train_output(key, key, value)
                else:
                    self.logger.error('Unknown stage found other than training or validation: %s' % (stage))
            self.logger.debug(message)
            return True

        # timeline trace saved
        if message.startswith('Timeline trace written to'):
            self.logger.info(message)
            self.detect_timeline_traces()
            return True

        # 此处为更新训练过程中保存pb文件的输出
        match = re.match(r'\S*Save intermediate result to\S*', message)
        if match:
            self.logger.info(message)
            self.detect_snapshots()
            self.send_snapshot_update()
            return True

        # network display starting
        if message.startswith('Network definition:'):
            self.displaying_network = True
            return True

        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        # skip remaining info and warn messages
        return True

    @staticmethod
    def preprocess_output_tensorflow(line):
        """
        获取retrain执行后的日志，用来处理进度条与时间
        """
        match = re.match(r'(\w+):tensorflow:(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{6}):\sStep\s(\d*):\s(\S.*)$', line)

        if match:
            level = match.group(1)
            timestamp = time.mktime(time.strptime(match.group(2)[:19], '%Y-%m-%d %H:%M:%S'))
            step = int(match.group(3))
            message = match.group(4)
            if level == 'INFO':
                level = 'info'
            elif level == 'DUBUG':
                level = 'debug'
            elif level == 'WARNING':
                level = 'warning'
            elif level == 'ERROR':
                level = 'error'
            elif level == 'FAIL':  # FAIL
                level = 'critical'
            return (timestamp, level, step, message)
        else:
            # self.logger.warning('Unrecognized task output "%s"' % line)
            return (None, None, None, None)

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        socketio.emit('task update', {'task': self.html_id(),
                                      'update': 'snapshots',
                                      'data': self.snapshot_list()},
                      namespace='/jobs',
                      room=self.job_id)

    # TrainTask overrides
    @override
    def after_run(self):
        if self.temp_unrecognized_output:
            if self.traceback:
                self.traceback = self.traceback + ('\n'.join(self.temp_unrecognized_output))
            else:
                self.traceback = '\n'.join(self.temp_unrecognized_output)
                self.temp_unrecognized_output = []
        self.tensorflow_log.close()

    @override
    def after_runtime_error(self):
        if os.path.exists(self.path(self.TENSORFLOW_LOG)):
            output = subprocess.check_output(['tail', '-n40', self.path(self.TENSORFLOW_LOG)])
            lines = []
            for line in output.split('\n'.encode('utf-8')):
                # parse tensorflow header
                timestamp, level, message = self.preprocess_output_tensorflow(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            traceback = '\n\nLast output:\n' + '\n'.join(lines[len(lines)-20:]) if len(lines) > 0 else ''
            if self.traceback:
                self.traceback = self.traceback + traceback
            else:
                self.traceback = traceback

            if 'DIGITS_MODE_TEST' in os.environ:
                print(output)

    @override
    def detect_timeline_traces(self):
        timeline_traces = []
        for filename in os.listdir(self.job_dir):
            # find timeline jsons
            match = re.match(r'%s_(.*)\.json$' % TIMELINE_PREFIX, filename)
            if match:
                step = int(match.group(1))
                timeline_traces.append((os.path.join(self.job_dir, filename), step))
        self.timeline_traces = sorted(timeline_traces, key=lambda tup: tup[1])
        return len(self.timeline_traces) > 0

    @override
    def detect_snapshots(self):
        # 获取模型目录下，根据步长间隔保存模型的数量，为根据步长间隔来选择不同阶段的模型进行操作
        self.snapshots = []
        snapshots = []
        for filename in os.listdir(self.job_dir):
            # find models
            # match = re.match(r'%s_(\d+)\.?(\d*)\.ckpt\.index$' % self.snapshot_prefix, filename)
            match = re.match(r'intermediate_graphintermediate_(\d*).pb$', filename)
            if match:
                step = 0
                # filename = filename[:-6]
                step = int(match.group(1))
                snapshots.append((os.path.join(self.job_dir, filename), step))
            if filename == 'frozen_model.pb':
                snapshots.append((os.path.join(self.job_dir, 'frozen_model.pb'), self.how_many_training_steps))
        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])
        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        # TODO: Currently this function is not in use. Probably in future we may have to implement this
        return None

    # 之后的代码全为弃用，hub模型预测调用DIGITS\digits\inference\tasks\hub_inference.py
    @override
    def infer_one(self,
                  data,
                  snapshot_epoch=None,
                  layers=None,
                  gpu=None,
                  resize=True):
        # resize parameter is unused
        return self.infer_one_image(data,
                                    snapshot_epoch=snapshot_epoch,
                                    layers=layers,
                                    gpu=gpu)

    def infer_one_image(self, image, snapshot_epoch=None, layers=None, gpu=None):
        """
        Classify an image
        Returns (predictions, visualizations)
            predictions -- an array of [ (label, confidence), ...] for each label, sorted by confidence
            visualizations -- an array of (layer_name, activations, weights) for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- a np.array

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        temp_image_handle, temp_image_path = tempfile.mkstemp(suffix='.tfrecords')
        os.close(temp_image_handle)
        if image.ndim < 3:
            image = image[..., np.newaxis]
        writer = tf.python_io.TFRecordWriter(temp_image_path)

        image = image.astype('float')
        record = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'image_raw': _float_array_feature(image.flatten()),
            'label': _int64_feature(0),
            'encoding': _int64_feature(0)}))
        writer.write(record.SerializeToString())
        writer.close()

        file_to_load = self.get_snapshot(snapshot_epoch)

        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'pb', 'label_image.py'),
                '--inference_db=%s' % temp_image_path,
                '--network=%s' % self.model_file,
                '--networkDirectory=%s' % self.job_dir,
                '--weights=%s' % file_to_load,
                '--allPredictions=1',
                '--batch_size=1',
                ]
        if hasattr(self.dataset, 'labels_file'):
            args.append('--labels_list=%s' % self.dataset.path(self.dataset.labels_file))

        if self.use_mean != 'none':
            mean_file = self.dataset.get_mean_file()
            assert mean_file is not None, 'Failed to retrieve mean file.'
            args.append('--mean=%s' % self.dataset.path(mean_file))

        if self.use_mean == 'pixel':
            args.append('--subtractMean=pixel')
        elif self.use_mean == 'image':
            args.append('--subtractMean=image')
        else:
            args.append('--subtractMean=none')

        if self.crop_size:
            args.append('--croplen=%d' % self.crop_size)

        if layers == 'all':
            args.append('--visualize_inf=1')
            args.append('--save=%s' % self.job_dir)

        # Convert them all to strings
        args = [str(x) for x in args]

        self.logger.info('%s classify one task started.' % self.get_framework_id())

        unrecognized_output = []
        predictions = []
        self.visualization_file = None

        env = os.environ.copy()

        if gpu is not None:
            # make only the selected GPU visible
            env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices([gpu])

        p = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             cwd=self.job_dir,
                             close_fds=True,
                             env=env)

        try:
            while p.poll() is None:
                for line in utils.nonblocking_readlines(p.stdout):
                    if self.aborted.is_set():
                        p.terminate()
                        raise digits.inference.errors.InferenceError(_('%(id)s classify one task got aborted. error code - %(returncode)d', id=self.get_framework_id(), returncode=p.returncode))  # noqa

                    if line is not None and len(line) > 1:
                        if not self.process_test_output(line, predictions, 'one'):
                            self.logger.warning('%s classify one task unrecognized input: %s' % (
                                self.get_framework_id(), line.strip()))
                            unrecognized_output.append(line)
                    else:
                        time.sleep(0.05)
        except Exception as e:
            if p.poll() is None:
                p.terminate()
            error_message = ''
            if type(e) == digits.inference.errors.InferenceError:
                error_message = e.__str__()
            else:
                error_message_log = '%s classify one task failed with error code %d \n %s' % (
                    self.get_framework_id(), p.returncode, str(e))
                error_message = _('%(id)s classify one task failed with error code %(returncode)d \n %(str_e)s',
                                  id=self.get_framework_id(), returncode=p.returncode, str_e=str(e))
            self.logger.error(error_message_log)
            if unrecognized_output:
                unrecognized_output = '\n'.join(unrecognized_output)
                error_message = error_message + unrecognized_output
            raise digits.inference.errors.InferenceError(error_message)

        finally:
            self.after_test_run(temp_image_path)

        if p.returncode != 0:
            error_message_log = '%s classify one task failed with error code %d' % (self.get_framework_id(), p.returncode)
            error_message = _('%(id)s classify one task failed with error code %(returncode)d', id=self.get_framework_id(), returncode=p.returncode)
            self.logger.error(error_message_log)
            if unrecognized_output:
                unrecognized_output = '\n'.join(unrecognized_output)
                error_message = error_message + unrecognized_output
            raise digits.inference.errors.InferenceError(error_message)
        else:
            self.logger.info('%s classify one task completed.' % self.get_framework_id())

        predictions = {'output': np.array(predictions)}

        visualizations = []

        if layers == 'all' and self.visualization_file:
            vis_db = h5py.File(self.visualization_file, 'r')
            # the HDF5 database is organized as follows:
            # <root>
            # |- layers
            #    |- 1
            #    |  [attrs] - op
            #    |  [attrs] - var
            #    |  |- activations
            #    |  |- weights
            #    |- 2
            for layer_id, layer in list(vis_db['layers'].items()):
                op_name = layer.attrs['op']
                var_name = layer.attrs['var']
                layer_desc = "%s\n%s" % (op_name, var_name)
                idx = int(layer_id)
                # activations (tf: operation outputs)
                if 'activations' in layer:
                    data = np.array(layer['activations'][...])
                    if len(data.shape) > 1 and data.shape[0] == 1:
                        # skip batch dimension
                        data = data[0]
                    if len(data.shape) == 3:
                        data = data.transpose(2, 0, 1)
                    elif len(data.shape) == 4:
                        data = data.transpose(3, 2, 0, 1)
                    vis = utils.image.get_layer_vis_square(data)
                    mean, std, hist = self.get_layer_statistics(data)
                    visualizations.append(
                        {
                            'id': idx,
                            'name': layer_desc,
                            'vis_type': 'Activations',
                            'vis': vis,
                            'data_stats': {
                                'shape': data.shape,
                                'mean':  mean,
                                'stddev':  std,
                                'histogram': hist,
                            }
                        }
                    )
                # weights (tf: variables)
                if 'weights' in layer:
                    data = np.array(layer['weights'][...])
                    if len(data.shape) == 3:
                        data = data.transpose(2, 0, 1)
                    elif len(data.shape) == 4:
                        data = data.transpose(3, 2, 0, 1)
                    if 'MatMul' in layer_desc:
                        vis = None  # too many layers to display?
                    else:
                        vis = utils.image.get_layer_vis_square(data)
                    mean, std, hist = self.get_layer_statistics(data)
                    parameter_count = reduce(operator.mul, data.shape, 1)
                    visualizations.append(
                        {
                            'id':  idx,
                            'name': layer_desc,
                            'vis_type': 'Weights',
                            'vis': vis,
                            'param_count': parameter_count,
                            'data_stats': {
                                'shape': data.shape,
                                'mean': mean,
                                'stddev': std,
                                'histogram': hist,
                            }
                        }
                    )
            # sort by layer ID
            visualizations = sorted(visualizations, key=lambda x: x['id'])
        return (predictions, visualizations)

    def get_layer_statistics(self, data):
        """
        Returns statistics for the given layer data:
            (mean, standard deviation, histogram)
                histogram -- [y, x, ticks]

        Arguments:
        data -- a np.ndarray
        """
        # These calculations can be super slow
        mean = np.mean(data)
        std = np.std(data)
        y, x = np.histogram(data, bins=20)
        y = list(y)
        ticks = x[[0, len(x)/2, -1]]
        x = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
        ticks = list(ticks)
        return (mean, std, [y, x, ticks])

    def after_test_run(self, temp_image_path):
        try:
            os.remove(temp_image_path)
        except OSError:
            pass

    def process_test_output(self, line, predictions, test_category):
        # parse torch output
        timestamp, level, message = self.preprocess_output_tensorflow(line)

        # return false when unrecognized output is encountered
        if not (level or message):
            return False

        if not message:
            return True

        float_exp = '([-]?inf|nan|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'

        # format of output while testing single image
        match = re.match(r'For image \d+, predicted class \d+: \d+ \((.*?)\) %s' % (float_exp), message)
        if match:
            label = match.group(1)
            confidence = match.group(2)
            assert not('inf' in confidence or 'nan' in confidence), 'Network reported %s for confidence value. Please check image and network' % label  # noqa
            confidence = float(confidence)
            predictions.append((label, confidence))
            return True

        # format of output while testing multiple images
        match = re.match(r'Predictions for image \d+: (.*)', message)
        if match:
            values = match.group(1).strip()
            # 'values' should contain a JSON representation of
            # the prediction
            predictions.append(eval(values))
            return True

        # path to visualization file
        match = re.match(r'Saving visualization to (.*)', message)
        if match:
            self.visualization_file = match.group(1).strip()
            return True

        # displaying info and warn messages as we aren't maintaining separate log file for model testing
        if level == 'info':
            self.logger.debug('%s classify %s task : %s' % (self.get_framework_id(), test_category, message))
            return True
        if level == 'warning':
            self.logger.warning('%s classify %s task : %s' % (self.get_framework_id(), test_category, message))
            return True

        if level in ['error', 'critical']:
            raise digits.inference.errors.InferenceError(_('%(id)s classify %(test_category)s task failed with error message - %(message)s',
                                                           id=self.get_framework_id(), test_category=test_category, message=message))

        return False  # control should never reach this line.

    @override
    def infer_many(self, data, snapshot_epoch=None, gpu=None, resize=True):
        # resize parameter is unused
        return self.infer_many_images(data, snapshot_epoch=snapshot_epoch, gpu=gpu)

    def infer_many_images(self, images, snapshot_epoch=None, gpu=None):
        """
        Returns (labels, results):
        labels -- an array of strings
        results -- a 2D np array:
            [
                [image0_label0_confidence, image0_label1_confidence, ...],
                [image1_label0_confidence, image1_label1_confidence, ...],
                ...
            ]

        Arguments:
        images -- a list of np.arrays

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        """

        # create a temporary folder to store images and a temporary file
        # to store a list of paths to the images
        temp_dir_path = tempfile.mkdtemp(suffix='.tfrecords')
        try:  # this try...finally clause is used to clean up the temp directory in any case
            with open(os.path.join(temp_dir_path, 'list.txt'), 'w') as imglist_file:
                for image in images:
                    if image.ndim < 3:
                        image = image[..., np.newaxis]
                    image = image.astype('float')
                    temp_image_handle, temp_image_path = tempfile.mkstemp(dir=temp_dir_path, suffix='.tfrecords')
                    writer = tf.python_io.TFRecordWriter(temp_image_path)
                    record = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(image.shape[0]),
                        'width': _int64_feature(image.shape[1]),
                        'depth': _int64_feature(image.shape[2]),
                        'image_raw': _float_array_feature(image.flatten()),
                        'label': _int64_feature(0),
                        'encoding': _int64_feature(0)}))
                    writer.write(record.SerializeToString())
                    writer.close()
                    imglist_file.write("%s\n" % temp_image_path)
                    os.close(temp_image_handle)

            file_to_load = self.get_snapshot(snapshot_epoch)

            args = [sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(digits.__file__)), 'tools', 'tensorflow', 'main.py'),
                    '--testMany=1',
                    '--allPredictions=1',  # all predictions are grabbed and formatted as required by DIGITS
                    '--inference_db=%s' % str(temp_dir_path),
                    '--network=%s' % self.model_file,
                    '--networkDirectory=%s' % self.job_dir,
                    '--weights=%s' % file_to_load,
                    ]

            if hasattr(self.dataset, 'labels_file'):
                args.append('--labels_list=%s' % self.dataset.path(self.dataset.labels_file))

            if self.use_mean != 'none':
                mean_file = self.dataset.get_mean_file()
                assert mean_file is not None, 'Failed to retrieve mean file.'
                args.append('--mean=%s' % self.dataset.path(mean_file))

            if self.use_mean == 'pixel':
                args.append('--subtractMean=pixel')
            elif self.use_mean == 'image':
                args.append('--subtractMean=image')
            else:
                args.append('--subtractMean=none')
            if self.crop_size:
                args.append('--croplen=%d' % self.crop_size)

            # Convert them all to strings
            args = [str(x) for x in args]

            self.logger.info('%s classify many task started.' % self.name())

            env = os.environ.copy()
            if gpu is not None:
                # make only the selected GPU visible
                env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices([gpu])

            unrecognized_output = []
            predictions = []
            p = subprocess.Popen(args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 cwd=self.job_dir,
                                 close_fds=True,
                                 env=env)

            try:
                while p.poll() is None:
                    for line in utils.nonblocking_readlines(p.stdout):
                        if self.aborted.is_set():
                            p.terminate()
                            raise digits.inference.errors.InferenceError(_('%(id)s classify many task got aborted.'
                                                                         'error code - %(returncode)d', id=self.get_framework_id(),
                                                                           returncode=p.returncode))

                        if line is not None and len(line) > 1:
                            if not self.process_test_output(line, predictions, 'many'):
                                self.logger.warning('%s classify many task unrecognized input: %s' % (
                                    self.get_framework_id(), line.strip()))
                                unrecognized_output.append(line)
                        else:
                            time.sleep(0.05)
            except Exception as e:
                if p.poll() is None:
                    p.terminate()
                error_message = ''
                if type(e) == digits.inference.errors.InferenceError:
                    error_message = e.__str__()
                else:
                    error_message_log = '%s classify many task failed with error code %d \n %s' % (
                        self.get_framework_id(), p.returncode, str(e))
                    error_message = _('%(id)s classify many task failed with error code %(returncode)d \n %(str_e)s',
                                      id=self.get_framework_id(), returncode=p.returncode, str_e=str(e))
                self.logger.error(error_message_log)
                if unrecognized_output:
                    unrecognized_output = '\n'.join(unrecognized_output)
                    error_message = error_message + unrecognized_output
                raise digits.inference.errors.InferenceError(error_message)

            if p.returncode != 0:
                error_message_log = '%s classify many task failed with error code %d' % (self.get_framework_id(),
                                                                                     p.returncode)
                error_message = _('%(id)s classify many task failed with error code %(returncode)d', id=self.get_framework_id(),
                                  returncode=p.returncode)
                self.logger.error(error_message_log)
                if unrecognized_output:
                    unrecognized_output = '\n'.join(unrecognized_output)
                    error_message = error_message + unrecognized_output
                raise digits.inference.errors.InferenceError(error_message)
            else:
                self.logger.info('%s classify many task completed.' % self.get_framework_id())
        finally:
            shutil.rmtree(temp_dir_path)

        # task.infer_one() expects dictionary in return value
        return {'output': np.array(predictions)}

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) != 0

    @override
    def get_model_files(self):
        """
        return paths to model files
        """
        return {"Network": 'checkpoint'}

    @override
    def get_network_desc(self):
        """
        return text description of network
        """
        with open(os.path.join(self.job_dir, TENSORFLOW_MODEL_FILE), "r") as infile:
            desc = infile.read()
        return desc

    @override
    def get_task_stats(self, epoch=-1):
        """
        return a dictionary of task statistics
        """

        loc, mean_file = os.path.split(self.dataset.get_mean_file())

        stats = {
            "image dimensions": self.dataset.get_feature_dims(),
            "mean file": mean_file,
            "snapshot file": self.get_snapshot_filename(epoch),
            "model file": self.model_file,
            "framework": "tensorflow_hub",
            "mean subtraction": self.use_mean,
        }

        if hasattr(self, "digits_version"):
            stats.update({"digits version": self.digits_version})

        if hasattr(self.dataset, "resize_mode"):
            stats.update({"image resize mode": self.dataset.resize_mode})

        if hasattr(self.dataset, "labels_file"):
            stats.update({"labels file": self.dataset.labels_file})

        return stats





