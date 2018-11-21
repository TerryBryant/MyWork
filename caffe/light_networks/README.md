### caffe实现的轻量级网络模型
1、```mobilenet_v2.py```，参考了ssd里面的basenet写成的，定义好invert_residual_block之后，不断堆叠就好了。
具体模型结构参考的是[这里](https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2_deploy.prototxt)的实现，
不知道为什么有的stride=1的block也没有shortcut，所以只有一行行的堆叠invert_residual_block

2、```mobilenet_v1.py```，具体模型结构参考的是[这里](https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_deploy.prototxt)的实现
