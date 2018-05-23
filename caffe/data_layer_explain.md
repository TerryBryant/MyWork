### 关于caffe数据层的参数说明
数据层是模型的入口，为模型提供数据以及标签，如果数据层有误，要么训练无法进行，要么训练无法收敛。下面具体记录下caffe的几种数据输入方式
#### 1、type:"Data"，数据来源于LevelDB或LMDB
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
    scale: 0.00390625    	                          // 像素值*scale*(1/256)
    mean_file: "examples/cifar10/mean.binaryproto"    // 用一个配置文件来进行均值操作，也可以用mean_value 
    mirror: ture                                      // ture表示开启镜像
    crop_size: 227                                    // 剪裁一个 227*227的图块，在训练阶段随机剪裁，在测试阶段从中间裁剪
  }
```
data_param中的backend默认为LevelDB，另外需要说明的是，在制作LMDB文件时已经指定了统一的图片尺寸（也可以统一mean_file）
#### 2、type:"ImageData"，数据来源于图片
```
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  image_data_param {
    source: "examples/_temp/train_list.txt"    // 存放训练集图片路径的txt
	root_folder: "examples/image/"             // 训练集图片路径
    batch_size: 50
    new_height: 256                            // 用于指定resize之后的尺寸，最好与crop_size一致
    new_width: 256
	shuffle: false                             // 默认值是false，表示每次epoch的是否不打乱数据集
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
}
```

