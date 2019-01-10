# coding: utf-8

from model.yolo_v3 import YOLO_V3
import config as cfg
from data import Data
import tensorflow as tf
import numpy as np
import os
import argparse
import logging
import time


class YoloTrain(object):
    def __init__(self):
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__learn_rate_init = cfg.LEARN_RATE_INIT
        self.__max_periods = cfg.MAX_PERIODS
        self.__max_wave_time = cfg.MAX_WAVE_TIME
        self.__max_learn_rate_decay_time = cfg.MAX_LEARN_RATE_DECAY_TIME
        self.__weights_dir = cfg.WEIGHTS_DIR
        self.__weights_file = cfg.WEIGHTS_FILE
        self.__time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'train', self.__time)
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__save_iter = cfg.SAVE_ITER
        self.__max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE

        self.__train_data = Data('train')
        self.__test_data = Data('test')

        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.__label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.__label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.__sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.__mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.__lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.__training = tf.placeholder(dtype=tf.bool, name='training')

        self.__yolo = YOLO_V3(self.__training)
        self.__conv_sbbox, self.__conv_mbbox, self.__conv_lbbox, \
        self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox = self.__yolo.build_nework(self.__input_data)
        self.__net_var = tf.global_variables()
        logging.info('Load weights:')
        for var in self.__net_var:
            logging.info(var.op.name)

        self.__loss = self.__yolo.loss(self.__conv_sbbox, self.__conv_mbbox, self.__conv_lbbox,
                                       self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox,
                                       self.__label_sbbox, self.__label_mbbox, self.__label_lbbox,
                                       self.__sbboxes, self.__mbboxes, self.__lbboxes)

        with tf.name_scope('learn'):
            self.__learn_rate = tf.Variable(self.__learn_rate_init, trainable=False, name='learn_rate_init')
            moving_ave = tf.train.ExponentialMovingAverage(self.__moving_ave_decay).apply(tf.trainable_variables())

            self.__trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.__trainable_var_list.append(var)
            optimize0 = tf.train.AdamOptimizer(self.__learn_rate).\
                minimize(self.__loss, var_list=self.__trainable_var_list)
            with tf.control_dependencies([optimize0]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op_with_frozen_variables = tf.no_op()

            optimize1 = tf.train.AdamOptimizer(self.__learn_rate).\
                minimize(self.__loss, var_list=tf.trainable_variables())
            with tf.control_dependencies([optimize1]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op_with_all_variables = tf.no_op()

            self.__train_op = self.__train_op_with_frozen_variables
            logging.info('Default trian step0 is freeze the weight of darknet')
            for var in self.__trainable_var_list:
                logging.info('\t' + str(var.op.name).ljust(50) + str(var.shape))

        with tf.name_scope('load_save'):
            self.__load = tf.train.Saver(self.__net_var)
            self.__save = tf.train.Saver(tf.global_variables(), max_to_keep=50)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.__loss)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())

        self.__sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def train(self, frozen=True):
        self.__sess.run(tf.global_variables_initializer())
        ckpt_path = os.path.join(self.__weights_dir, self.__weights_file)
        logging.info('Restoring weights from:\t %s' % ckpt_path)
        self.__load.restore(self.__sess, ckpt_path)

        learn_rate_decay_time = 0
        test_loss_err_list = []
        test_loss_last = np.inf
        for period in range(self.__max_periods):
            wave_time = (np.array(test_loss_err_list) > 0).astype(np.int32).sum()
            if frozen and wave_time == self.__max_wave_time:
                test_loss_err_list = []
                test_loss_last = np.inf
                if learn_rate_decay_time < self.__max_learn_rate_decay_time:
                    learning_rate_value = self.__sess.run(
                        tf.assign(self.__learn_rate, self.__sess.run(self.__learn_rate) / 10.0)
                    )
                    logging.info('The value of learn rate is:\t%f' % learning_rate_value)

                # 使用原始learn rate_init * 0.01微调至饱和后再用learn_rate_init * 0.01全部微调
                learn_rate_decay_time += 1
                if learn_rate_decay_time == (self.__max_learn_rate_decay_time + 1):
                    self.__train_op = self.__train_op_with_all_variables
                    logging.info('Train all of weights')
                    self.__train_data.batch_size_change(6)
                    self.__test_data.batch_size_change(6)

            if not frozen:
                self.__train_op = self.__train_op_with_all_variables
                logging.info('Train all of weights')

            print_loss_iter = len(self.__train_data) / 10
            total_train_loss = 0.0

            for step, (batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                       batch_sbboxes, batch_mbboxes, batch_lbboxes) \
                    in enumerate(self.__train_data):
                _, summary_value, loss_value = self.__sess.run(
                    [self.__train_op, self.__summary_op, self.__loss],
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_sbbox: batch_label_sbbox,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__sbboxes: batch_sbboxes,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: False
                    }
                )
                print "keep running"
                if np.isnan(loss_value):
                    raise ArithmeticError('The gradient is exploded')
                total_train_loss += loss_value
                if (step + 1) % print_loss_iter:
                    continue
                train_loss = total_train_loss / print_loss_iter
                total_train_loss = 0.0
                self.__summary_writer.add_summary(summary_value, period * len(self.__train_data) + step)
                logging.info('Period:\t%d\tstep:\t%d\ttrain loss:\t%.4f' % (period, step, train_loss))

            if (period + 1) % self.__save_iter:
                continue

            total_test_loss = 0.0
            for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                batch_sbboxes, batch_mbboxes, batch_lbboxes \
                    in self.__test_data:
                loss_value = self.__sess.run(
                    self.__loss,
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_sbbox: batch_label_sbbox,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__sbboxes: batch_sbboxes,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: False
                }
                )
                print "keep running"
                total_test_loss += loss_value
            test_loss = total_test_loss / len(self.__test_data)
            logging.info('Period:\t%d\ttest loss:\t%.4f' % (period, test_loss))
            saved_model_name = os.path.join(self.__weights_dir, 'yolo.ckpt-%d-%.4f' % (period, test_loss))
            self.__save.save(self.__sess, saved_model_name)
            logging.info('Saved model:\t%s' % saved_model_name)

            test_loss_err_list.append(test_loss - test_loss_last)
            test_loss_last = test_loss
        self.__summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', default='yolo_coco_initial.ckpt', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default='32', type=str)
    parser.add_argument('--frozen', default='True', type=str)
    parser.add_argument('--learn_rate_init', default='0.001', type=str)
    args = parser.parse_args()

    log_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logging.basicConfig(filename='log/train/' + log_time + '.log', format='%(filename)s %(asctime)s\t%(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')

    if args.gpu is not None:
        cfg.GPU = args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    if args.weights_file is not None:
        cfg.WEIGHTS_FILE = args.weights_file
    cfg.BATCH_SIZE = int(args.batch_size)
    logging.info('Batch size is:\t%d' % cfg.BATCH_SIZE)
    cfg.LEARN_RATE_INIT = float(args.learn_rate_init)
    logging.info('Initial learn rate is:\t%f' % cfg.LEARN_RATE_INIT)
    T = YoloTrain()
    assert args.frozen in ['True', 'False']
    if args.frozen == 'True':
        T.train(True)
    else:
        T.train(False)

