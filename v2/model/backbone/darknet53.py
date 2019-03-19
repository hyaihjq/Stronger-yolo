# coding: utf-8

import sys
import os
sys.path.append(os.path.abspath('../..'))
from model.layers import *


def darknet53(input_data, training):
    with tf.variable_scope('darknet53'):
        with tf.variable_scope('stage0'):
            conv = convolutional(name='conv0', input_data=input_data, filters_shape=(3, 3, 3, 32),
                                 training=training)
            conv = convolutional(name='conv1', input_data=conv, filters_shape=(3, 3, 32, 64),
                                 training=training, downsample=True)

        with tf.variable_scope('stage1'):
            conv = residual_block(name='residual0', input_data=conv, input_channel=64,
                                  filter_num1=32, filter_num2=64, training=training)
            conv = convolutional(name='conv0', input_data=conv, filters_shape=(3, 3, 64, 128),
                                 training=training, downsample=True)

        with tf.variable_scope('stage2'):
            conv = residual_block(name='residual0', input_data=conv, input_channel=128,
                                  filter_num1=64, filter_num2=128, training=training)
            conv = residual_block(name='residual1', input_data=conv, input_channel=128,
                                  filter_num1=64, filter_num2=128, training=training)
            conv = convolutional(name='conv0', input_data=conv, filters_shape=(3, 3, 128, 256),
                                 training=training, downsample=True)

        with tf.variable_scope('stage3'):
            conv = residual_block(name='residual0', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual1', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual2', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual3', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual4', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual5', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            conv = residual_block(name='residual6', input_data=conv, input_channel=256,
                                  filter_num1=128, filter_num2=256, training=training)
            darknet_route0 = residual_block(name='residual7', input_data=conv, input_channel=256,
                                            filter_num1=128, filter_num2=256, training=training)
            conv = convolutional(name='conv0', input_data=darknet_route0, filters_shape=(3, 3, 256, 512),
                                 training=training, downsample=True)

        with tf.variable_scope('stage4'):
            conv = residual_block(name='residual0', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual1', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual2', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual3', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual4', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual5', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            conv = residual_block(name='residual6', input_data=conv, input_channel=512,
                                  filter_num1=256, filter_num2=512, training=training)
            darknet_route1 = residual_block(name='residual7', input_data=conv, input_channel=512,
                                            filter_num1=256, filter_num2=512, training=training)
            conv = convolutional(name='conv0', input_data=darknet_route1, filters_shape=(3, 3, 512, 1024),
                                 training=training, downsample=True)

        with tf.variable_scope('stage5'):
            conv = residual_block(name='residual0', input_data=conv, input_channel=1024,
                                  filter_num1=512, filter_num2=1024, training=training)
            conv = residual_block(name='residual1', input_data=conv, input_channel=1024,
                                  filter_num1=512, filter_num2=1024, training=training)
            conv = residual_block(name='residual2', input_data=conv, input_channel=1024,
                                  filter_num1=512, filter_num2=1024, training=training)
            darknet_route2 = residual_block(name='residual3', input_data=conv, input_channel=1024,
                                            filter_num1=512, filter_num2=1024, training=training)
        return darknet_route0, darknet_route1, darknet_route2