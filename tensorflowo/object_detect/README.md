## tensorflow目标检测
从tensorflow官方找的目标检测例子，自行下载相关的模型文件，完成inference
### 流程
1、自行下载[模型文件](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)以及相应的[label文件](https://github.com/tensorflow/models/tree/master/research/object_detection/data)  
2、下载[辅助文件夹](https://github.com/tensorflow/models/tree/master/research/object_detection)，这是一个文件夹，可以通过这个叫做[DownGit](https://minhaskamal.github.io/DownGit/#/home)的网站来下载单个文件夹，这个文件夹有一百多M，实际上只需要用protos和utils两个文件夹，但protos文件夹下的部分文件需要通过protoc来生成，不想逐个文件排查，就把整个object_detection文件夹下载了，然后执行
```
protoc object_detection/protos/*.proto --python_out=.
```
即在object_detection/protos下生成了必须要的文件，注意protoc需要3.0以上的版本，否则会报错  
3、准备好图片文件，执行```ssd_mobilenet_inference.py```，即可得到检测结果（该脚本参考的是[这里](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)，终于体会到python为什么会被人黑游标卡尺了。。）
