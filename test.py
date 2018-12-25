# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os
import tensorflow as tf
from model.yolo_v3 import YOLO_V3
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import shutil
import random
import colorsys
import argparse
from utils import utils


class YoloTest(object):
    def __init__(self):
        self.__test_input_size = cfg.TEST_INPUT_SIZE
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))
        self.__anchors = np.array(cfg.ANCHORS)
        self.__score_threshold = cfg.SCORE_THRESHOLD
        self.__iou_threshold = cfg.IOU_THRESHOLD
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'test')
        self.__annot_dir_path = cfg.ANNOT_DIR_PATH
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__dataset_path = cfg.DATASET_PATH

        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__training = tf.placeholder(dtype=tf.bool, name='training')
        _, _, _, self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox = \
            YOLO_V3(self.__training).build_nework(self.__input_data)
        with tf.name_scope('summary'):
            tf.summary.FileWriter(self.__log_dir).add_graph(tf.get_default_graph())
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.__moving_ave_decay)
        self.__sess = tf.Session()
        self.__saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.__saver.restore(self.__sess, os.path.join(cfg.WEIGHTS_DIR, cfg.WEIGHTS_FILE))

    def __get_bbox(self, image):
        """
        :param image: 要预测的图片
        :return: 返回NMS后的bboxes，存储格式为(xmin, ymin, xmax, ymax, score, class)
        """
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        yolo_input = utils.img_preprocess2(image, None, (self.__test_input_size, self.__test_input_size), False)
        yolo_input = yolo_input[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.__sess.run(
            [self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox],
            feed_dict={
                self.__input_data: yolo_input,
                self.__training:False
            }
        )

        sbboxes = self.__convert_pred(pred_sbbox, (org_h, org_w))
        mbboxes = self.__convert_pred(pred_mbbox, (org_h, org_w))
        lbboxes = self.__convert_pred(pred_lbbox, (org_h, org_w))
        bboxes = np.concatenate([sbboxes, mbboxes, lbboxes], axis=0)
        bboxes = utils.nms(bboxes,self.__score_threshold, self.__iou_threshold, method='nms')
        return bboxes

    def __convert_pred(self, pred_bbox, org_img_shape):
        """
        将yolo输出的bbox信息(x, y, w, h, confidence, probability)进行转换，
        其中(x, y, w, h)是预测bbox的中心坐标、宽、高，大小是相对于input_size的，
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        (x, y, w, h) --> (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
        --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
        :param pred_bbox: yolo输出的bbox信息，shape为(1, output_size, output_size, anchor_per_scale, 5 + num_classes)
        :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
        :return: bboxes
        假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        """
        pred_bbox = np.array(pred_bbox)
        output_size = pred_bbox.shape[1]
        pred_bbox = np.reshape(pred_bbox, (output_size, output_size, self.__anchor_per_scale, 5 + self.__num_classes))
        pred_xywh = pred_bbox[:, :, :, 0:4]
        pred_conf = pred_bbox[:, :, :, 4:5]
        pred_prob = pred_bbox[:, :, :, 5:]

        # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :, :, :2] - pred_xywh[:, :, :, 2:] * 0.5,
                                    pred_xywh[:, :, :, :2] + pred_xywh[:, :, :, 2:] * 0.5], axis=-1)
        # (2)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * self.__test_input_size / org_w, 1.0 * self.__test_input_size / org_h)
        dw = (self.__test_input_size - resize_ratio * org_w) / 2
        dh = (self.__test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, :, :, 0::2] = 1.0 * (pred_coor[:, :, :, 0::2] - dw) / resize_ratio
        pred_coor[:, :, :, 1::2] = 1.0 * (pred_coor[:, :, :, 1::2] - dh) / resize_ratio

        # (3)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :, :, :2], [0, 0]),
                                    np.minimum(pred_coor[:, :, :, 2:], [org_w - 1, org_h - 1])], axis=-1)

        pred_coor = pred_coor.reshape((-1, 4))
        pred_conf = pred_conf.reshape((-1,))
        pred_prob = pred_prob.reshape((-1,self.__num_classes))

        # (4)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.__score_threshold
        coors = pred_coor[score_mask]
        scores = scores[score_mask]
        classes = classes[score_mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def detect_image(self, image):
        original_image = np.copy(image)
        bboxes = self.__get_bbox(image)
        image = self.__draw_bbox(original_image, bboxes)
        return image

    def mAP(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path):
            shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path):
            shutil.rmtree(ground_truth_dir_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)

        annotation_path = os.path.join(self.__annot_dir_path, 'test_annotation.txt')
        with file(annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([map(int, box.split(',')) for box in annotation[1:]])
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print 'ground truth of %s:' % image_name
                num_bbox_gt = len(bboxes_gt)
                with file(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.__classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = map(str, bboxes_gt[i])
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print '\t' + str(bbox_mess).strip()
                print 'predict result of %s:' % image_name
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                bboxes_pr = self.__get_bbox(image)
                with file(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.__classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = map(str, coor)
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print '\t' + str(bbox_mess).strip()
                print

    def voc_2012_test(self):
        test_2012_path = os.path.join(self.__dataset_path, '2012_test')
        img_inds_file = os.path.join(test_2012_path, 'ImageSets', 'Main',  'test.txt')
        with file(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

        for image_ind in image_inds:
            image_path = os.path.join(test_2012_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print 'predict result of %s:' % image_ind
            bboxes_pr = self.__get_bbox(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.__classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with file(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
                print '\t' + str(bbox_mess).strip()
            print
        pass

    def __draw_bbox(self, original_image, bboxes):
        """
        :param original_image: 检测的原始图片，shape为(org_h, org_w, 3)
        :param bboxes: shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        :return: None
        """
        hsv_tuples = [(1.0 * x / self.__num_classes, 1., 1.) for x in range(self.__num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        image_h, image_w, _ = original_image.shape
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(1.0 * (image_h + image_w) / 600)
            cv2.rectangle(original_image, (coor[0], coor[1]), (coor[2], coor[3]), bbox_color, bbox_thick)

            bbox_mess = '%s: %.3f' % (self.__classes[class_ind], score)
            text_loc = (int(coor[0]), int(coor[1] + 5) if coor[1] < 20 else int(coor[1] - 5))
            cv2.putText(original_image, bbox_mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h, (255, 255, 255), bbox_thick // 3)
        return original_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_calc', default='False', type=str)
    parser.add_argument('--test_2012', default='False', type=str)
    parser.add_argument('--weights_file', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    map_calc = args.map_calc
    test_2012 = args.test_2012
    weights_file = args.weights_file
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cfg.WEIGHTS_FILE = weights_file
    weights_path = os.path.join(cfg.WEIGHTS_DIR, cfg.WEIGHTS_FILE) + '.index'
    if not os.path.exists(weights_path):
        raise RuntimeError(str('You must enter a valid name of weights in directory: %s' % cfg.WEIGHTS_DIR))

    T = YoloTest()
    if map_calc == 'True':
        T.mAP()
    elif test_2012 == 'True':
        T.voc_2012_test()
    else:
        images = ['./data/' + image for image in os.listdir('./data')
                  if (image[-3:] == 'jpg') and (image[0] != '.') ]
        image = cv2.imread(np.random.choice(images))
        image = T.detect_image(image)
        cv2.imwrite('detect_result.jpg', image)


