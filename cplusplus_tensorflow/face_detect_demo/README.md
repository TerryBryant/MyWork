### 人脸检测API
参考的是[这里](https://github.com/yeephycho/tensorflow-face-detection)，是一个基于tensorflow ssd mobilenet finetune的人脸检测模型。效果不错，速度也还行，虽然比不了mtcnn（inference太难写），比opencv自带的人脸检测还是准多了。


```face_detect.cpp```是根据python inference代码整理成的c++代码，这里直接借用了对方的模型文件，并对检测到的人脸进行了扩展，将来可以集成到其它项目里了

```function_samples.cpp```包括了几个利用tensorflow c++ api写的函数，以后可以作为参考
