Note
=
**I'm solving scale invariant. If you have a good paper, you can email me by StinkyTofu95@gmail.com. Thanks!**<br>

## Improve yolo_v3 with latest paper <br>
#### performance on VOC2007(Better performance than [Tencent's](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo) reimplementation)<br>
<table>
   <tr><td>model</td><td>initial with yolov3.weights</td><td></td><td>initial with darknet.weights</td><td></td><td>release</td></tr>
   <tr><td></td><td>mAP</td><td>delta</td><td>mAP</td><td>delta</td><td></td></tr>
   <tr><td>baseline</td><td>84.3</td><td>0.0</td><td>72.3</td><td>0.0</td><td>yes</td></tr>
   <tr><td>data agumentation</td><td>85.8</td><td>+1.5</td><td>75.9</td><td>+3.6</td><td>yes</td></tr>
   <tr><td>multi scale train</td><td>86.3</td><td>+0.5</td><td>78.3</td><td>+2.4</td><td>yes</td></tr>
   <tr><td>focal loss</td><td>88.3</td><td>+2.0</td><td>79.6</td><td>+1.3</td><td>yes</td></tr>
   <tr><td>des</td><td>xxx</td><td>+1.0</td><td>xxx</td><td>xxx</td><td>no</td></tr>
   <tr><td>group normalization</td><td>xxx</td><td>xxx</td><td>xxx</td><td>xxx</td><td>yes</td></tr>
   <tr><td>soft nms</td><td>xxx</td><td>-0.6</td><td>xxx</td><td>-0.6</td><td>yes</td></tr>
   <tr><td>modify positive and negative labels</td><td>88.9</td><td>+0.6</td><td>79.3</td><td>-0.3</td><td>yes</td></tr>
   <tr><td>mix up</td><td>xxx</td><td>-0.3</td><td>80.7</td><td>+1.4</td><td>no</td></tr>
   <tr><td>label smooth</td><td>89.1</td><td>+0.2</td><td>xxx</td><td>xxx</td><td>yes</td></tr>
   <tr><td>multi scale test</td><td>90.7</td><td>+1.6</td><td>82.8</td><td>+2.1</td><td>yes</td></tr>
</table>
<p align="center">evaluated at 544x544 on Pascal VOC 2007 test set</p>

![mAP](https://github.com/Stinky-Tofu/Stronger-yolo/blob/master/mAP/mAP0.png)<br>

![mAP](https://github.com/Stinky-Tofu/Stronger-yolo/blob/master/mAP/mAP1.png)<br>
    
#### to do
- [ ] Deformable convolutional networks<br>
- [ ] Scale-Aware Trident Networks for Object Detection
- [ ] Understanding the Effective Receptive Field in Deep Convolutional Neural Networks<br>

## Usage
1. clone YOLO_v3 repository
    ``` bash
    git clone https://github.com/Stinky-Tofu/Stronger-yolo.git
    ```
2. prepare data<br>
    (1) download datasets<br>
    Create a new folder named `data` in the directory where the `YOLO_V3` folder 
    is located, and then create a new folder named `VOC` in the `data/`.<br>
    Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and put datasets into `data/VOC`,
    name as `2012_trainval`、`2007_trainval`、`2007_test` separately. <br>
    The file structure is as follows:<br>
    |--YOLO_V3<br>
    |--data<br>
    |--|--VOC<br>
    |--|--|--2012_trainval<br>
    |--|--|--2007_trainval<br>
    |--|--|--2007_test<br>
    (2) convert data format<br>
    You should set `DATASET_PATH` in `config.py` to the path of the VOC dataset, for example:
    `DATASET_PATH = '/home/xzh/doc/code/python_code/data/VOC'`,and then<br>
    ```bash
    python voc_annotation.py
    ```
3. prepare initial weights<br>
    Download [YOLOv3-608.weights](https://pjreddie.com/media/files/yolov3.weights) firstly, 
    put the yolov3.weights into `yolov3_to_tf/`, and then 
    ```bash
    cd yolov3_to_tf
    python3 convert_weights.py --weights_file=yolov3.weights --dara_format=NHWC --ckpt_file=./saved_model/yolov3_608_coco_pretrained.ckpt
    cd ..
    python rename.py
    ``` 

4. Train<br>
    ``` bash
    python train.py
    ```
5. Test<br>
    Download weight file [yolo_test.ckpt](https://drive.google.com/drive/folders/1We_P5L4nlLofR0IJJXzS7EEklZGUb9sz)<br>
    **If you want to get a higher mAP, you can set the score threshold to 0.01、use multi scale test、flip test.<br>
    If you want to use it in actual projects, or if you want speed, you can set the score threshold to 0.2.<br>**
    ``` bash
    python test.py --gpu=0 --map_calc=True --weights_file=model_path.ckpt
    cd mAP
    python main.py -na -np
    ```
## Train for custom dataset<br>

1. Generate your own annotation file `train_annotation.txt` 
and `test_annotation.txt`, one row for one image. <br>
Row format: image_path bbox0 bbox1 ...<br>
Bbox format: xmin,ymin,xmax,ymax,class_id(no space), for example:<br>
    ```bash
    /home/xzh/doc/code/python_code/data/VOC/2007_test/JPEGImages/000001.jpg 48,240,195,371,11 8,12,352,498,14
    ```
2. Put the `train_annotation.txt` and `test_annotation.txt` into `YOLO_V3/data/`.<br>
3. Configure config.py for your dataset.<br>
3. Start training.<br>
    ```bash
    python train.py
    ```
     
## Reference:<br>
paper: <br>
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)<br>
- [Foca Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)<br>
- [Group Normalization](https://arxiv.org/abs/1803.08494)<br>
- [Single-Shot Object Detection with Enriched Semantics](https://arxiv.org/abs/1712.00433)<br>
- [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)<br>
- [Deformable convolutional networks](https://arxiv.org/abs/1811.11168)<br>
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)<br>
- [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)<br>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)<br>

mAP calculate: [mean Average Precision](https://github.com/Cartucho/mAP)<br>
 
## Requirements
- Python2.7.12 <br>
- Numpy1.14.5<br>
- Tensorflow.1.8.0 <br>
- Opencv3.4.1 <br>
