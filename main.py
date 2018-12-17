# coding: utf-8

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    data = tf.placeholder(tf.float32)
    data_shape = tf.shape(data)
    output_size = data_shape[1]
    batch_size = data_shape[0]

    reshaped_data = tf.reshape(data, (data_shape[0], data_shape[1], data_shape[2], 3, 3))


    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])


    with tf.Session() as sess:
        data_val, arr_val = sess.run([reshaped_data, xy_grid], feed_dict={
            data: np.zeros((2, 7, 7, 9))
        })
        print data_val.shape
        print arr_val