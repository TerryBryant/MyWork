### 关于caffe数据层的参数说明
数据层是模型的入口，为模型提供数据以及标签，如果数据层有误，要么训练无法进行，要么训练无法收敛。下面具体记录下caffe的几种数据输入方式
#### 1、type:"Data"
```
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
```
说明下这里面的transform_param，主要涉及到数据预处理，示例如下
```
transform_param {
    scale: 0.00390625
    mean_file_size: "examples/cifar10/mean.binaryproto"
	[comment]: <> (This is a comment, it will not be included)
	[comment]: <> (in  the output file unless you use it in)
	[comment]: <> (a reference style link.)
	[//]: # (This may be the most platform independent comment)
    # 用一个配置文件来进行均值操作
    mirror: 1  # 1表示开启镜像，0表示关闭，也可用ture和false来表示
    # 剪裁一个 227*227的图块，在训练阶段随机剪裁，在测试阶段从中间裁剪
    crop_size: 227
  }
```
