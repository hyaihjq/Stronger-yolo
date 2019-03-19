# coding: utf-8

import numpy as np
import os
import config as cfg
from utils import utils
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import cv2
from utils import data_aug
import random
import tensorflow as tf
import logging


class Data(object):
    def __init__(self, dataset_type, split_ratio=1.0):
        """
        需始终记住：
        small_detector对应下标索引0， medium_detector对应下标索引1，big_detector对应下标索引2
        :param dataset_type: 选择加载训练样本或测试样本，必须是'train' or 'test'
        """
        self.__annot_dir_path = cfg.ANNOT_DIR_PATH
        self.__train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self.__strides = np.array(cfg.STRIDES)
        self.__batch_size = cfg.BATCH_SIZE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__anchors = np.array(cfg.ANCHORS)
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))
        self.__max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE

        annotations = self.__load_annotations(dataset_type)
        num_annotations = len(annotations)
        self.__annotations = annotations[: int(split_ratio * num_annotations)]
        self.__num_samples = len(self.__annotations)
        logging.info(('The number of image for %s is:' % dataset_type).ljust(50) + str(self.__num_samples))
        self.__num_batchs = np.ceil(self.__num_samples / self.__batch_size)
        self.__batch_count = 0

    def batch_size_change(self, batch_size_new):
        self.__batch_size = batch_size_new
        self.__num_batchs = np.ceil(self.__num_samples / self.__batch_size)
        logging.info('Use the new batch size: %d' % self.__batch_size)


    def __load_annotations(self, dataset_type):
        """
        :param dataset_type: 选择加载训练样本或测试样本，必须是'train' or 'test'
        :return: annotations，每个元素的形式如下：
        image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...
        """
        if dataset_type not in ['train', 'test']:
            raise ImportError("You must choice one of the 'train' or 'test' for dataset_type parameter")
        annotation_path = os.path.join(self.__annot_dir_path, dataset_type + '_annotation.txt')
        with file(annotation_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def next(self):
        """
        使得pascal_voc对象变为可迭代对象
        :return: 每次迭代返回一个batch的图片、标签
        batch_image: shape为(batch_size, input_size, input_size, 3)
        batch_label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, 5 + num_classes)
        batch_label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, 5 + num_classes)
        batch_label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, 5 + num_classes)
        batch_sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        batch_mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        batch_lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        """
        with tf.device('/cpu:0'):
            if (self.__batch_count % 10) == 0:
                self.__train_input_size = random.choice(self.__train_input_sizes)
                self.__train_output_sizes = self.__train_input_size / self.__strides

            batch_image = np.zeros((self.__batch_size, self.__train_input_size, self.__train_input_size, 3))
            batch_label_sbbox = np.zeros((self.__batch_size, self.__train_output_sizes[0], self.__train_output_sizes[0],
                                          self.__anchor_per_scale, 5 + self.__num_classes))
            batch_label_mbbox = np.zeros((self.__batch_size, self.__train_output_sizes[1], self.__train_output_sizes[1],
                                          self.__anchor_per_scale, 5 + self.__num_classes))
            batch_label_lbbox = np.zeros((self.__batch_size, self.__train_output_sizes[2], self.__train_output_sizes[2],
                                          self.__anchor_per_scale, 5 + self.__num_classes))
            batch_sbboxes = np.zeros((self.__batch_size, self.__max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.__batch_size, self.__max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.__batch_size, self.__max_bbox_per_scale, 4))
            num = 0
            if self.__batch_count < self.__num_batchs:
                while num < self.__batch_size:
                    index = self.__batch_count * self.__batch_size + num
                    if index >= self.__num_samples:
                        index -= self.__num_samples
                    annotation = self.__annotations[index]
                    image, bboxes = self.__parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__create_label(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.__batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.__batch_count = 0
                np.random.shuffle(self.__annotations)
                raise StopIteration

    def __parse_annotation(self, annotation):
        """
        读取annotation中image_path对应的图片，并将该图片进行resize(不改变图片的高宽比)
        获取annotation中所有的bbox，并将这些bbox的坐标(xmin, ymin, xmax, ymax)进行纠正，
        使得纠正后bbox在resize后的图片中的相对位置与纠正前bbox在resize前的图片中的相对位置相同
        :param annotation: 图片地址和bbox的坐标、类别，
        如：image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...
        :return: image和bboxes
        bboxes的shape为(N, 5)，其中N表示一站图中有N个bbox，5表示(xmin, ymin, xmax, ymax, class_ind)
        """
        line = annotation.split()
        image_path = line[0]
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([map(int, box.split(',')) for box in line[1:]])

        image, bboxes = data_aug.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = data_aug.random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = data_aug.random_translate(np.copy(image), np.copy(bboxes))
        image, bboxes = utils.img_preprocess2(np.copy(image), np.copy(bboxes),
                                              (self.__train_input_size, self.__train_input_size), True)
        return image, bboxes

    def __create_label(self, bboxes):
        """
        (1.25, 1.625), (2.0, 3.75), (4.125, 2.875) 这三个anchor用于small_detector预测小物体
        [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375) 这三个anchor用于medium_detector预测中物体
        [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875) 这三个anchor用于big_detector预测大物体
        与bbox有最大IOU的anchor，视为best anchor，best anchor所属的detector负责预测该bbox，
        根据这一准则，对于一张图的所有bbox，每个detector都有自己要负责预测的bbox
        small_detector负责预测的bbox放在sbboxes中，
        medium_detector负责预测的bbox放在mbboxes中
        big_detector负责预测的bbox放在lbboxes中
        需始终记住：
        small_detector对应下标索引0， medium_detector对应下标索引1，big_detector对应下标索引2
        :param bboxes: 一张图对应的所有bbox和每个bbox所属的类别，bbox的坐标为(xmin, ymin, xmax, ymax, class_ind)
        :return:
        label_sbbox: shape为(input_size / 8, input_size / 8, anchor_per_scale, 5 + num_classes)
        label_mbbox: shape为(input_size / 16, input_size / 16, anchor_per_scale, 5 + num_classes)
        label_lbbox: shape为(input_size / 32, input_size / 32, anchor_per_scale, 5 + num_classes)
        只有best anchor对应位置的数据才为(x, y, w, h, 1, classes), (x, y, w, h)的大小是bbox纠正后的原始大小
        其他非best anchor对应位置的数据都为(0, 0, 0, 0, 0, 0...)
        sbboxes：shape为(max_bbox_per_scale, 4)
        mbboxes：shape为(max_bbox_per_scale, 4)
        lbboxes：shape为(max_bbox_per_scale, 4)
        存储的坐标为(x, y, w, h)，(x, y, w, h)的大小都是bbox纠正后的原始大小
        bboxes用于计算相应detector的预测框与该detector负责预测的所有bbox的IOU
        """
        label = [np.zeros((self.__train_output_sizes[i], self.__train_output_sizes[i], self.__anchor_per_scale,
                           5 + self.__num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.__max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            # label smooth
            onehot = np.zeros(self.__num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.__num_classes, 1.0 / self.__num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # (1)(xmin, ymin, xmax, ymax) -> (x, y, w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            # (2)(x, y, w, h) / stride
            # 对bbox使用三种stride，得到三个detector对应的尺度
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.__strides[:, np.newaxis]

            # (3)计算所有Anchor与该bbox的IOU，并获取最大IOU对应的best anchor
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.__anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.__anchors[i]

                iou_scale = utils.iou_calc2(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # (4)将iou大于0.3的anchor对应位置的数据标识为(x, y, w, h, 1, classes)
                    # 首先需要将该Anchor对应的标签清零，因为某个Anchor可能与多个bbox的IOU大于0.3
                    # 如果不清零，那么该Anchor可能会被标记为多类
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.__max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.__anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.__anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # (4)将best_anchor对应位置的数据标识为(x, y, w, h, 1, classes)
                # 首先需要将该Anchor对应的标签清零，因为某个Anchor可能与多个bbox有最大IOU，
                # 当输入图片尺寸为416时，与多个bbox有最大IOU的Anchor总共有248个
                # 如果不清零，那么该Anchor可能会被标记为多类
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.__max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.__num_batchs


if __name__ == '__main__':
    data_obj = Data('train')
    for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in data_obj:
        print batch_image.shape
        print batch_label_sbbox.shape
        print batch_label_mbbox.shape
        print batch_label_lbbox.shape
        print batch_sbboxes.shape
        print batch_mbboxes.shape
        print batch_lbboxes.shape

    data_obj = Data('test')
    for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in data_obj:
        print batch_image.shape
        print batch_label_sbbox.shape
        print batch_label_mbbox.shape
        print batch_label_lbbox.shape
        print batch_sbboxes.shape
        print batch_mbboxes.shape
        print batch_lbboxes.shape
