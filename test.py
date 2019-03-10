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
        self.__train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self.__multi_test = cfg.MULTI_TEST
        self.__flip_test = cfg.FLIP_TEST
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
        if self.__multi_test:
            test_input_sizes = self.__train_input_sizes[::3]
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(self.__predict(image, test_input_size, valid_scale))
                if self.__flip_test:
                    bboxes_flip = self.__predict(image[:, ::-1, :], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = image.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(image, self.__test_input_size, (0, np.inf))
        bboxes = utils.nms(bboxes, self.__score_threshold, self.__iou_threshold, method='nms')
        return bboxes

    def __predict(self, image, test_input_size, valid_scale):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        yolo_input = utils.img_preprocess2(image, None, (test_input_size, test_input_size), False)
        yolo_input = yolo_input[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = self.__sess.run(
            [self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox],
            feed_dict={
                self.__input_data: yolo_input,
                self.__training: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.__num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.__num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.__num_classes))], axis=0)
        bboxes = self.__convert_pred(pred_bbox, test_input_size, (org_h, org_w), valid_scale)
        return bboxes

    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        将yolo输出的bbox信息(x, y, w, h, confidence, probability)进行转换，
        其中(x, y, w, h)是预测bbox的中心坐标、宽、高，大小是相对于input_size的，
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        (x, y, w, h) --> (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
        --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
        :param pred_bbox: yolo输出的bbox信息，shape为(output_size * output_size * anchor_per_scale, 5 + num_classes)
        :param test_input_size: 测试尺寸
        :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
        :return: bboxes
        假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        """
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # (2)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (3)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (4)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (4)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.__score_threshold

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

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
                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
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
        img_inds_file = os.path.join(test_2012_path, 'ImageSets', 'Main', 'test.txt')
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
    def str2bool(v):
        if v.lower() in ('true'):
            return True
        elif v.lower() in ('false'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_test', default=False, type=str2bool)
    parser.add_argument('--flip_test', default=False, type=str2bool)
    parser.add_argument('--map_calc', default=False, type=str2bool)
    parser.add_argument('--test_2012', default=False, type=str2bool)
    parser.add_argument('--weights_file', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    cfg.MULTI_TEST = args.multi_test
    cfg.FLIP_TEST = args.flip_test
    map_calc = args.map_calc
    test_2012 = args.test_2012
    weights_file = args.weights_file
    cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    cfg.WEIGHTS_FILE = weights_file
    weights_path = os.path.join(cfg.WEIGHTS_DIR, cfg.WEIGHTS_FILE) + '.index'
    if not os.path.exists(weights_path):
        raise RuntimeError(str('You must enter a valid name of weights in directory: %s' % cfg.WEIGHTS_DIR))

    T = YoloTest()
    if map_calc:
        T.mAP()
    elif test_2012:
        T.voc_2012_test()
    else:
        images = ['./data/' + image for image in os.listdir('./data/test_data')
                  if (image[-3:] == 'jpg') and (image[0] != '.')]
        image = cv2.imread(np.random.choice(images))
        image = T.detect_image(image)
        cv2.imwrite('detect_result.jpg', image)


