# Initial with YOLOV3-608.weights
## Performance<br>
train dataset: VOC 2012 + VOC 2007<br>
test dataset: VOC 2007<br>
test size: 544<br>
test code: [mAP](https://github.com/Cartucho/mAP) (not use 07 metric)<br>
test score threshold: 0.01<br>
<table>
   <tr><td>model</td><td>mAP</td><td>delta</td><td>release</td></tr>
   <tr><td>baseline</td><td>84.3</td><td>0.0</td><td>yes</td></tr>
   <tr><td>data agumentation</td><td>85.8</td><td>+1.5</td><td>yes</td></tr>
   <tr><td>multi scale train</td><td>86.3</td><td>+0.5</td><td>yes</td></tr>
   <tr><td>focal loss</td><td>88.3</td><td>+2.0</td><td>yes</td></tr>
   <tr><td>des</td><td>xxx</td><td>+1.0</td><td>no</td></tr>
   <tr><td>group normalization</td><td>xxx</td><td>xxx</td><td>yes</td></tr>
   <tr><td>soft nms</td><td>xxx</td><td>-0.6</td><td>yes</td></tr>
   <tr><td>label smooth</td><td>88.6</td><td>+0.3</td><td>yes</td></tr>
   <tr><td>cosine learning rate</td><td>88.6</td><td>0.0</td><td>yes</td></tr>
   <tr><td>GIOU</td><td>88.8</td><td>+0.2</td><td>yes</td></tr>
   <tr><td>multi scale test</td><td>90.7</td><td>+1.9</td><td>yes</td></tr>
</table>

## Usage
1. clone YOLO_v3 repository
    ``` bash
    git clone https://github.com/Stinky-Tofu/Stronger-yolo.git
    ```
2. prepare data<br>
    (1) download datasets<br>
    Create a new folder named `data` in the directory where the `Stronger-yolo` folder 
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
    You should alter `DATASET_PATH` in `config.py`, for example:<br>
    `DATASET_PATH = /home/wz/doc/code/python_code/data/VOC`<br>
    ```bash
    cd Stronger-yolo/v1
    python voc_annotation.py
    ```
3. prepare initial weights<br>
    Download [YOLOv3-608.weights](https://pjreddie.com/media/files/yolov3.weights) firstly, 
    put the yolov3.weights into `yolov3_to_tf/`, and then 
    ```bash
    cd yolov3_to_tf
    python3 convert_weights.py --weights_file=yolov3.weights --data_format=NHWC --ckpt_file=./saved_model/yolov3_608_coco_pretrained.ckpt
    cd ..
    python rename.py
    ``` 

4. train<br>
    ``` bash
    python train.py
    ```
5. test<br>
    Download weight file [stronger-yolo-v1-test.ckpt](https://drive.google.com/drive/folders/1We_P5L4nlLofR0IJJXzS7EEklZGUb9sz)<br>
    **If you want to get a higher mAP, you can set the score threshold to 0.01、use multi scale test、flip test.<br>
    If you want to use it in actual projects, or if you want speed, you can set the score threshold to 0.2.<br>**
    ``` bash
    python test.py --gpu=0 --map_calc=True --weights_file=model_path.ckpt
    cd mAP
    python main.py -na -np
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
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)<br>
 
## Requirements
software
- Python2.7.12 <br>
- Numpy1.14.5<br>
- Tensorflow.1.8.0 <br>
- Opencv3.4.1 <br>

hardware
- 12G 1080Ti
