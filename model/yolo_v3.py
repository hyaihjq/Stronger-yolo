# coding:utf-8

import tensorflow as tf
import custom_layers as cs_layer
import config as cfg
import numpy as np
from utils import utils


class YOLO_V3(object):
    def __init__(self, training):
        self.__training = training
        self.__classes = cfg.CLASSES
        self.__num_classes = len(cfg.CLASSES)
        self.__strides = np.array(cfg.STRIDES)
        self.__anchors = np.array(cfg.ANCHORS)
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__iou_loss_thresh = cfg.IOU_LOSS_THRESH

    def build_nework(self, input_data):
        """
        :param input_data: shape为(batch_size, input_size, input_size, 3)
        :return: conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
        conv_sbbox的shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale * (5 + num_classes))
        conv_mbbox的shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale * (5 + num_classes))
        conv_lbbox的shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale * (5 + num_classes))
        conv_?是YOLO的原始卷积输出(raw_dx, raw_dy, raw_dw, raw_dh, raw_conf, raw_prob)
        pred_sbbox的shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale, 5 + num_classes)
        pred_mbbox的shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale, 5 + num_classes)
        pred_lbbox的shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale, 5 + num_classes)
        pred_?是YOLO预测bbox的信息(x, y, w, h, conf, prob)，(x, y, w, h)的大小是相对于input_size的
        """
        with tf.variable_scope('darknet'):
            conv = cs_layer.convolutional(name='conv0', input_data=input_data, filters_shape=(3, 3, 3, 32),
                                          training=self.__training)
            conv = cs_layer.convolutional(name='conv1', input_data=conv, filters_shape=(3, 3, 32, 64),
                                          training=self.__training, downsample=True)

            # residual block part0
            conv = cs_layer.residual_block(name='residual0', input_data=conv, input_channel=64,
                                           filter_num1=32, filter_num2=64, training=self.__training)
            conv = cs_layer.convolutional(name='conv4', input_data=conv, filters_shape=(3, 3, 64, 128),
                                          training=self.__training, downsample=True)

            # residual block part1
            conv = cs_layer.residual_block(name='residual1', input_data=conv, input_channel=128,
                                           filter_num1=64, filter_num2=128, training=self.__training)
            conv = cs_layer.residual_block(name='residual2', input_data=conv, input_channel=128,
                                           filter_num1=64, filter_num2=128, training=self.__training)
            conv = cs_layer.convolutional(name='conv9', input_data=conv, filters_shape=(3, 3, 128, 256),
                                          training=self.__training, downsample=True)

            # residual block part2
            conv = cs_layer.residual_block(name='residual3', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256, training=self.__training)
            conv = cs_layer.residual_block(name='residual4', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256,
                                           training=self.__training)
            conv = cs_layer.residual_block(name='residual5', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256,
                                           training=self.__training)
            conv = cs_layer.residual_block(name='residual6', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256,
                                           training=self.__training)
            conv = cs_layer.residual_block(name='residual7', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256,
                                           training=self.__training)
            conv = cs_layer.residual_block(name='residual8', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256,
                                           training=self.__training)
            conv = cs_layer.residual_block(name='residual9', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256, training=self.__training)
            darknet_route1 = cs_layer.residual_block(name='residual10', input_data=conv, input_channel=256,
                                           filter_num1=128, filter_num2=256, training=self.__training)
            conv = cs_layer.convolutional(name='conv26', input_data=darknet_route1, filters_shape=(3, 3, 256, 512),
                                          training=self.__training, downsample=True)

            # redidual block part3
            conv = cs_layer.residual_block(name='residual11', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual12', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual13', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual14', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual15', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual16', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.residual_block(name='residual17', input_data=conv, input_channel=512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            darknet_route2 = cs_layer.residual_block(name='residual18', input_data=conv, input_channel= 512,
                                           filter_num1=256, filter_num2=512, training=self.__training)
            conv = cs_layer.convolutional(name='conv43', input_data=darknet_route2, filters_shape=(3, 3, 512, 1024),
                                          training=self.__training, downsample=True)

            # residual block part4
            conv = cs_layer.residual_block(name='residual19', input_data=conv, input_channel=1024,
                                           filter_num1=512, filter_num2=1024, training=self.__training)
            conv = cs_layer.residual_block(name='residual20', input_data=conv, input_channel=1024,
                                           filter_num1=512, filter_num2=1024, training=self.__training)
            conv = cs_layer.residual_block(name='residual21', input_data=conv, input_channel=1024,
                                           filter_num1=512, filter_num2=1024, training=self.__training)
            conv = cs_layer.residual_block(name='residual22', input_data=conv, input_channel=1024,
                                           filter_num1=512, filter_num2=1024, training=self.__training)

        # conv经过几个卷积层之后作为检测分支的输入，这几个卷积层不改变输入conv的的shape
        conv = cs_layer.convolutional(name='conv52', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv53', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv54', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv55', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv56', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                      training=self.__training)

        # ----------**********---------- Detection branch of large object ----------**********----------
        conv_lobj_branch = cs_layer.convolutional(name='conv_lobj_branch', input_data=conv,
                                                  filters_shape=(3, 3, 512, 1024), training=self.__training)
        conv_lbbox = cs_layer.convolutional(name='conv_lbbox', input_data=conv_lobj_branch,
                                            filters_shape=(1, 1, 1024, self.__anchor_per_scale * (self.__num_classes + 5)),
                                            training=self.__training, downsample=False, activate=False, bn=False)
        pred_lbbox = cs_layer.decode(name='pred_lbbox', conv_output=conv_lbbox, anchors=self.__anchors[2],
                                     num_classes=self.__num_classes, stride=self.__strides[2])

        # ----------**********---------- Detection branch of large object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        # up sample之前用1x1的卷积将conv的channel变为256，以与darknet_route2的channel匹配
        conv = cs_layer.convolutional(name='conv57', input_data=conv, filters_shape=(1, 1, 512, 256),
                                      training=self.__training)
        conv = cs_layer.upsample(name='upsample0', input_data=conv)
        conv = cs_layer.route(name='route0', previous_output=darknet_route2, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        # up sample后的conv经过几个卷积层之后作为检测分支的输入，这几个卷积层不改变输入conv的的shape
        conv = cs_layer.convolutional('conv58', input_data=conv, filters_shape=(1, 1, 512+256, 256),
                                      training=self.__training)
        conv = cs_layer.convolutional('conv59', input_data=conv, filters_shape=(3, 3, 256, 512),
                                      training=self.__training)
        conv = cs_layer.convolutional('conv60', input_data=conv, filters_shape=(1, 1, 512, 256),
                                      training=self.__training)
        conv = cs_layer.convolutional('conv61', input_data=conv, filters_shape=(3, 3, 256, 512),
                                      training=self.__training)
        conv = cs_layer.convolutional('conv62', input_data=conv, filters_shape=(1, 1, 512, 256),
                                      training=self.__training)


        # ----------**********---------- Detection branch of middle object ----------**********----------
        conv_mobj_branch = cs_layer.convolutional(name='conv_mobj_branch', input_data=conv,
                                                  filters_shape=(3, 3, 256, 512), training=self.__training)
        conv_mbbox = cs_layer.convolutional(name='conv_mbbox', input_data=conv_mobj_branch,
                                            filters_shape=(1, 1, 512, self.__anchor_per_scale * (self.__num_classes + 5)),
                                            training=self.__training, downsample=False, activate=False, bn=False)
        pred_mbbox = cs_layer.decode(name='pred_mbbox', conv_output=conv_mbbox, anchors=self.__anchors[1],
                                     num_classes=self.__num_classes, stride=self.__strides[1])
        # ----------**********---------- Detection branch of middle object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        # up sample之前用1x1的卷积将conv的channel变为128，以与darknet_route2的channel匹配
        conv = cs_layer.convolutional(name='conv63', input_data=conv, filters_shape=(1, 1, 256, 128),
                                      training=self.__training)
        conv = cs_layer.upsample(name='upsample1', input_data=conv)
        conv = cs_layer.route(name='route1', previous_output=darknet_route1, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        # up sample后的conv经过几个卷积层之后作为检测分支的输入，这几个卷积层不改变输入conv的的shape
        conv = cs_layer.convolutional(name='conv64', input_data=conv, filters_shape=(1, 1, 256+128, 128),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv65', input_data=conv, filters_shape=(3, 3, 128, 256),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv66', input_data=conv, filters_shape=(1, 1, 256, 128),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv67', input_data=conv, filters_shape=(3, 3, 128, 256),
                                      training=self.__training)
        conv = cs_layer.convolutional(name='conv68', input_data=conv, filters_shape=(1, 1, 256, 128),
                                      training=self.__training)

        # ----------**********---------- Detection branch of small object ----------**********----------
        conv_sobj_branch = cs_layer.convolutional(name='conv_sobj_branch', input_data=conv,
                                                  filters_shape=(3, 3, 128, 256), training=self.__training)
        conv_sbbox = cs_layer.convolutional(name='conv_sbbox', input_data=conv_sobj_branch,
                                            filters_shape=(1, 1, 256, self.__anchor_per_scale * (self.__num_classes + 5)),
                                            training=self.__training, downsample=False, activate=False, bn=False)
        pred_sbbox = cs_layer.decode(name='pred_sbbox', conv_output=conv_sbbox, anchors=self.__anchors[0],
                                     num_classes=self.__num_classes, stride=self.__strides[0])
        # ----------**********---------- Detection branch of small object ----------**********----------

        return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox

    def __focal(self, target, actual, alpha=1, gamma=2):
        focal = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal

    def __loss_per_scale(self, name, conv, pred, label, bboxes, anchors, stride):
        """
        :param name: loss的名字
        :param conv: conv是yolo卷积层的原始输出
        shape为(batch_size, output_size, output_size, anchor_per_scale * (5 + num_class))
        :param pred: conv是yolo输出的预测bbox的信息(x, y, w, h, conf, prob)，
        其中(x, y, w, h)的大小是相对于input_size的，如input_size=416，(x, y, w, h) = (120, 200, 50, 70)
        shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_class)
        :param label: shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes)
        只有best anchor对应位置的数据才为(x, y, w, h, 1, classes), (x, y, w, h)的大小是bbox纠正后的原始大小
        :param bboxes: shape为(batch_size, max_bbox_per_scale, 4)，
        存储的坐标为(x, y, w, h)，(x, y, w, h)的大小都是bbox纠正后的原始大小
        bboxes用于计算相应detector的预测框与该detector负责预测的所有bbox的IOU
        :param anchors: 相应detector的anchors
        :param stride: 相应detector的stride
        """
        with tf.name_scope(name):
            conv_shape = tf.shape(conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]
            input_size = stride * output_size
            conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                     self.__anchor_per_scale, 5 + self.__num_classes))
            conv_raw_conf = conv[:, :, :, :, 4:5]
            conv_raw_prob = conv[:, :, :, :, 5:]

            pred_xywh = pred[:, :, :, :, 0:4]
            pred_conf = pred[:, :, :, :, 4:5]

            label_xywh = label[:, :, :, :, 0:4]
            respond_bbox = label[:, :, :, :, 4:5]
            label_prob = label[:, :, :, :, 5:]

            GIOU = utils.GIOU(pred_xywh, label_xywh)
            GIOU = GIOU[..., np.newaxis]
            input_size = tf.cast(input_size, tf.float32)
            bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
            GIOU_loss = respond_bbox * bbox_loss_scale * (1.0 - GIOU)

            # (2)计算confidence损失
            iou = utils.iou_calc4(pred_xywh[:, :, :, :, np.newaxis, :],
                                  bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, : ])
            max_iou = tf.reduce_max(iou, axis=-1)
            max_iou = max_iou[:, :, :, :, np.newaxis]
            respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.__iou_loss_thresh, tf.float32)

            conf_focal = self.__focal(respond_bbox, pred_conf)

            conf_loss = conf_focal * (
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                    +
                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            )

            # (3)计算classes损失
            prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
            loss = tf.concat([GIOU_loss, conf_loss, prob_loss], axis=-1)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))
            return loss

    def loss(self,
             conv_sbbox, conv_mbbox, conv_lbbox,
             pred_sbbox, pred_mbbox, pred_lbbox,
             label_sbbox, label_mbbox, label_lbbox,
             sbboxes, mbboxes, lbboxes):
        """
        :param conv_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale * (5 + num_classes))
        :param conv_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale * (5 + num_classes))
        :param conv_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale * (5 + num_classes))
        :param pred_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale, (5 + num_classes))
        :param pred_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale, (5 + num_classes))
        :param pred_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale, (5 + num_classes))
        :param label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale, 5 + num_classes)
        :param label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale, 5 + num_classes)
        :param label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale, 5 + num_classes)
        :param sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :param mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :param lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :return:
        """
        loss_sbbox = self.__loss_per_scale('loss_sbbox', conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                           self.__anchors[0], self.__strides[0])
        loss_mbbox = self.__loss_per_scale('loss_mbbox', conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                           self.__anchors[1], self.__strides[1])
        loss_lbbox = self.__loss_per_scale('loss_lbbox', conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                           self.__anchors[2], self.__strides[2])
        with tf.name_scope('loss'):
            loss = loss_sbbox + loss_mbbox + loss_lbbox
        return loss
