### 解析caffemodel，需要结合相应的prototxt文件来进行
需要注意的几点是:

1、BatchNorm层具有weight和bias两项参数，也就相当于pytorch里面的running_mean, running_var

2、Convolution层，有的时候bias_term=false（偏置项合并到后面bn层去了），此时注意Convolution层只有一项weight参数
