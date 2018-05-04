### opencv dnn模块读取tensorflow模型
- ```mnist_cnn.py```用于训练mnist数据，得到tensorflow模型，即```frozen_model.pb```。网络为最简单的lenet-5模型，保存pb文件的时候，需要
指定output_node_names，本模型文件中，输入输出都只有一个变量，所以opencv在加载的时候就自动识别了。但如果模型有多个输入，就得注意net.setInput()的
写法了


- ```python_tensorflow.py```根据```frozen_model.pb```来加载模型，代码中既有opencv dnn模块读取模型，也有tensorflow读取模型，可以看到两种方法
得到的结果是一样的


- ```opencv_tensorflow.cpp```是opencv dnn模块读取模型的c++代码，结果输出跟前面的一致
### PS
几个月前做opencv读取tensorflow模型的时候，opencv3.4还没出来，只能用opencv3.3，结果3.3相当不成熟，经常莫名其妙报一些错，坑了好久。今天把opencv更新到3.4.1，发现dnn模块还是稳定了不少，目前基本的网络层支持的都还不错，所以强烈推荐把opencv更新到最新的版本
