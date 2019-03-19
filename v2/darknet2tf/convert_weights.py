# -*- coding: utf-8 -*-

import tensorflow as tf

from yolo_v3 import darknet

from utils import load_coco_names, load_weights

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'weights_file', '', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/darknet53.ckpt', 'Chceckpoint file')


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])

    darknet(inputs, data_format=FLAGS.data_format)

    with tf.name_scope('summary'):
        summary_writer = tf.summary.FileWriter('./log')
        summary_writer.add_graph(tf.get_default_graph())

    load_ops = load_weights(tf.global_variables(scope='darknet'), FLAGS.weights_file)
    saver = tf.train.Saver(tf.global_variables(scope='darknet'))

    with tf.Session() as sess:
        sess.run(load_ops)

        save_path = saver.save(sess, save_path=FLAGS.ckpt_file)
        print('Model saved in path: {}'.format(save_path))

