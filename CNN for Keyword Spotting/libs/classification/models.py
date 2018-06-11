"""Model definitions for simple speech recognition.
    create_conv_model: Basic Convolutional Model
    
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from . import resnet_model
from .inception.slim import inception_model


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count,use_spectrogram=False):
    """Calculates common settings needed for all models.
  
    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.
  
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None,model_size_info=None):
    """Builds a model of the requested architecture compatible with the settings.
  
    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].
  
    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.
  
    See the implementations below for the possible model architectures that can be
    requested.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      model_architecture: String specifying which kind of model to create.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
  
    Raises:
      Exception: If the architecture type isn't recognized.
    """
    elif model_architecture == 'ds_cnn':
        return create_ds_cnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
  
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

    

def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info,
                        is_training):
    """Builds a model with depthwise separable convolutional neural network
    Model definition is based on https://arxiv.org/abs/1704.04861 and
    Tensorflow implementation: https://github.com/Zehaos/MobileNet
    model_size_info: defines number of layers, followed by the DS-Conv layer
      parameters in the order {number of conv features, conv filter height, 
      width and stride in y,x dir.} for each of the layers. 
    Note that first layer is always regular convolution, but the remaining 
      layers are all depthwise separable convolutions.
    """

    def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
  
    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers
    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.96,
                                updates_collections=None,
                                activation_fn=tf.nn.relu):
                for layer_no in range(0, num_layers):
                    if layer_no == 0:
                        net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                 [conv_kt[layer_no], conv_kf[layer_no]],
                                                 stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME',
                                                 scope='conv_1')
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    else:
                        net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                        kernel_size=[conv_kt[layer_no], conv_kf[layer_no]], \
                                                        stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                        sc='conv_ds_' + str(layer_no))
                    t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                    f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

                net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

    if is_training:
        return logits, dropout_prob
    else:
        return logits

