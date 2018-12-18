### caffe下的ssd相关代码
修改了一个基于resnet18的ssd模型，修改的方法其实很简单，因为ssd的[作者本身](https://github.com/weiliu89/caffe/tree/ssd)提供了基于resnet101的模型，所以只需要参考该模型修改部分代码即可。但由于对resnet的结构理解不够深刻，导致了自己还是犯了一些错。最主要的就是，resnet18、34结构比较接近，而resnet50、101、152结构比较接近。所以基于作者提供的resnet101来改一个resnet50的模型很容易，改几个地方就好了，而resnet18需要改的地方较多。另外在网上搜了一圈，也没有发现基于resnet18的ssd，有点意外，所以这里记录一下是怎么做的吧。

1、改写```python/caffe/model_libs.py```。这个文件用于生成vgg、resnet101等basenet，在```examples/ssd/ssd_pascal_resnet.py```中会调用它，具体就是```ResNet101Body```这个函数。可以直接修改```model_libs.py```文件，改好之后重新编译caffe的python接口。这里为了简便，我新建一个```model_libs_resnet.py```，将生成resnet18的相关代码写进去了，具体参考resnet18的网络结构，以及```model_libs.py```中的网络模型写法。

2、改写```examples/ssd/ssd_pascal_resnet.py```。为了方便，这里我也是新建了一个```ssd_resnet18.py```文件，文件内容跟```ssd_pascal_resnet.py```一致，再改写下```mbox_source_layers = ['res3b1_relu', 'res5b_relu', 'res5b_relu/conv1_2', 'res5b_relu/conv2_2', 'res5b_relu/conv3_2', 'pool6']```，以及改变两处```ResNet101Body```为```ResNet18Body```，再将第1步中的```model_libs_resnet.py```移动到这个目录下即可

3、配置好```ssd_resnet18.py```的相关路径，运行即可得到网络文件，这里统一上传到```resnet18_ssd```目录下

### 2018-11-19 update
1、注意在resnet作为基础网络的时候，从res4b1到res5a之后，按道理feature map只有10x10的尺寸了。在原始ssd中是这么处理的，
将这里卷积的stride从2变为1，然后把从这之后的卷积都改为dilation convolution，相当于也在保持感受野尺寸。所以在修改其它网络结构的时候，
建议也参考这种改法。

2、加入mobilenet v1基础网络。

3、注意在修改```mbox_source_layers```的时候，最后一个pool层的名字和basenet的最后一个pool层名字不同，不然会混淆

### 2018-11-23 update
加入densenet121基础网络，注意在使用dilation后，可能造成concat层的两个输入不匹配，此时应调整相应的padding


### 2018-12-18 update
有关dilation的问题，我之前还是没有理解清楚，后面去查了资料，现在更新一下。根据公式：output = (input - k + 2 * p - (k-1)(d-1)) / s + 1，
其中k是卷积核尺寸，p是padding，d是dilation尺寸，s是stride，input和output是输入输出尺寸。现在已知的是，stride从2变成了1（主要是在basenet
的最后几层，特征图尺寸不能再减小了），此时dilation应该变为2，为了保持输入输出尺寸一致，根据以上公式可以得到：p = k - 1

另外值得注意的是，如果你需要使用tensorrt对ssd进行加速，那么需要将dilation改为1（相应调整padding），目前它还不支持dilation大于1的卷积
