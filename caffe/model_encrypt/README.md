### caffe模型文件加密
为了防止模型文件被人使用，可以采用AES加密方式对caffe模型进行加密。具体方法也很简单，将deploy.prototxt或是deploy.caffemodel文件读入内存，
再利用AES对其中的字符进行加密，将加密后的字符重新写出来，即可得到加密后的模型文件了。关键就在于key的设置，一定要保密。
