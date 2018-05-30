import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


file_prototxt = '/home/terry/tmp/vgg/deploy.prototxt'
file_model = '/home/terry/tmp/vgg/vgg_remove_the_last_fc.caffemodel'

# # 通过修改deploy.prototxt文件（删除某些层），可以得到新的caffemodel，其中的权重与当前的deploy.prototxt一致
# # 可用于预训练的vgg finetune到ssd上，因为要删除vgg的最后两个全连接层
# net = caffe.Net(file_prototxt, file_model, caffe.TEST)
# net.save('vgg_remove_the_last_fc.caffemodel')
#
# for param_name in net.params.keys():
#     print(param_name)




with open(file_prototxt) as f:
    str = f.read()

msg = caffe_pb2.NetParameter()
text_format.Merge(str, msg)

net = caffe.Net(file_prototxt, file_model, caffe.TEST)

for i, layer in enumerate(msg.layer):
    layer_name = msg.layer[i].name
    layer_type = msg.layer[i].type

    # 激活和池化层没有参数（Dropout层也没有）
    if layer_type == 'ReLU' or layer_type == 'Pooling':
        continue

    shape = net.params[layer_name][0].data.shape

    weights = net.params[layer_name][0].data
    biases = net.params[layer_name][1].data

    print(layer_name)
    print(layer_type)
    print(shape)
    print('-------------------------')




