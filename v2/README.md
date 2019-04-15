# Initial with Darknet53-448.weights
## Performance(Better performance than [Tencent's](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo) reimplementation)<br>
train dataset: VOC 2012 + VOC 2007<br>
test dataset: VOC 2007<br>
test size: 544<br>
test code: [Faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py) (not use 07 metric)<br>
test score threshold: 0.01<br>

<table>
   <tr><td>model</td><td>mAP</td><td>delta</td><td>release</td></tr>
   <tr><td>baseline</td><td>73.3</td><td>0.0</td><td>yes</td></tr>
   <tr><td>data agumentation</td><td>76.9</td><td>+3.6</td><td>yes</td></tr>
   <tr><td>multi scale train</td><td>79.3</td><td>+2.4</td><td>yes</td></tr>
   <tr><td>focal loss</td><td>80.6</td><td>+1.3</td><td>yes</td></tr>
   <tr><td>group normalization</td><td>xxx</td><td>-0.5</td><td>yes</td></tr>
   <tr><td>soft nms</td><td>xxx</td><td>-0.6</td><td>yes</td></tr>
   <tr><td>mix up</td><td>81.7</td><td>+1.1</td><td>yes</td></tr>
   <tr><td>label smooth</td><td>82.1</td><td>+0.4</td><td>yes</td></tr>
   <tr><td>cosine learning rate</td><td>83.1</td><td>+1.0</td><td>yes</td></tr>
   <tr><td>GIOU</td><td>83.3</td><td>+0.2</td><td>yes</td></tr>
   <tr><td>remove anchor</td><td>83.3</td><td>0</td><td>yes</td></tr>
   <tr><td>multi scale test</td><td>85.8</td><td>2.5</td><td>yes</td></tr>
</table>

## Usage
1. clone YOLO_v3 repository
    ``` bash
    git clone https://github.com/Stinky-Tofu/Stronger-yolo.git
    ```
2. prepare data<br>
    (1) download datasets<br>
    Create a new folder named `data` in the directory where the `stronger-yolo` folder 
    is located, and then create a new folder named `VOC` in the `data/`.<br>
    Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and put datasets into `data/VOC`,
    name as `2012_trainval`、`2007_trainval`、`2007_test` separately. <br>
    The file structure is as follows:<br>
    |--stronger-yolo<br>
    |--|--v1<br>
    |--|--v2<br>
    |--|--v3<br>
    |--data<br>
    |--|--VOC<br>
    |--|--|--2012_trainval<br>
    |--|--|--2007_trainval<br>
    |--|--|--2007_test<br>
    (2) convert data format<br>
    You should alter `DATASET_PATH` and `PROJECT_PATH`in `config.py`, for example:<br>
    `DATASET_PATH = /home/wz/doc/code/python_code/data/VOC`<br>
    `PROJECT_PATH = /home/wz/doc/code/python_code/Stronger-yolo/v2`<br>
    and then<br>
    ```bash
    cd Stronger-yolo/v2/utils
    python voc.py
    ```
3. prepare initial weights<br>
    Download [darknet53_448.weights](https://pjreddie.com/media/files/darknet53_448.weights) firstly, 
    put the initial weights into `darknet2tf/`, and then 
    ```bash
    cd darknet2tf
    python3 convert_weights.py --weights_file=darknet53_448.weights --data_format=NHWC
    ``` 

4. train<br>
    ``` bash
    nohup python train.py &
    ```
5. test<br>
    Download weight file [stronger-yolo-v2-test.ckpt](https://drive.google.com/drive/folders/1HOwQ7RBefHrPDzYY3rlOWW1qiJ_7X7xz)<br>
    **If you want to get a higher mAP, you can set the score threshold to 0.01、use multi scale test、flip test.<br>
    If you want to use it in actual projects, or if you want speed, you can set the score threshold to 0.2.<br>**
    ``` bash
    nohup python test.py --gpu=0 --test_weight=model_path.ckpt -t07 &
    ```
     
## Reference:<br>
paper: <br>
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)<br>
- [Foca Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)<br>
- [Group Normalization](https://arxiv.org/abs/1803.08494)<br>
- [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)<br>
- [Deformable convolutional networks](https://arxiv.org/abs/1811.11168)<br>
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)<br>
- [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)<br>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)<br>
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)<br>
- [Stonger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo)<br>
 
## Requirements
software
- Python2.7.12 <br>
- Numpy1.14.5<br>
- Tensorflow.1.8.0 <br>
- Opencv3.4.1 <br>

hardware
- 16G 1080Ti
