### caffe下的ssd相关代码
修改了一个基于resnet18的ssd模型，修改的方法其实很简单，因为ssd的[作者本身](https://github.com/weiliu89/caffe/tree/ssd)提供了基于resnet101的模型，所以只需要参考该模型修改部分代码即可。但由于对resnet的结构理解不够深刻，导致了自己还是犯了一些错。最主要的就是，resnet18、34结构比较接近，而resnet50、101、152结构比较接近。所以基于作者提供的resnet101来改一个resnet50的模型很容易，改几个地方就好了，而resnet18需要改的地方较多。另外在网上搜了一圈，也没有发现基于resnet18的ssd，有点意外，所以这里记录一下是怎么做的吧。

1、改写```python/caffe/model_libs.py```。这个文件用于生成vgg、resnet101等basenet，在```examples/ssd/ssd_pascal_resnet.py```中会调用它，具体就是```ResNet101Body```这个函数。可以直接修改```model_libs.py```文件，改好之后重新编译caffe的python接口。这里为了简便，我新建一个```model_libs_resnet.py```，将生成resnet18的相关代码写进去了，具体参考resnet18的网络结构，以及```model_libs.py```中的网络模型写法。

2、改写```examples/ssd/ssd_pascal_resnet.py```。为了方便，这里我也是新建了一个```ssd_resnet18.py```文件，文件内容跟```ssd_pascal_resnet.py```一致，再改写下```mbox_source_layers = ['res3b1_relu', 'res5b_relu', 'res5b_relu/conv1_2', 'res5b_relu/conv2_2', 'res5b_relu/conv3_2', 'pool6']```，以及改变两处```ResNet101Body```为```ResNet18Body```，再将第1步中的```model_libs_resnet.py```移动到这个目录下即可

3、配置好```ssd_resnet18.py```的相关路径，运行即可得到网络文件，这里统一上传
