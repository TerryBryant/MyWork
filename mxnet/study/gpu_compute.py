import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

# mx.cpu()，无论括号里面填什么，都代表所有的物理CPU和内存
# mx.gpu()等价于mx.gpu(0)


x = nd.array([1, 2, 3])
# print(x.context)    # 查看x所在的设备（CPU）


# 在GPU上的操作
a = nd.array([1, 2, 3], ctx=mx.gpu())
# print(a)


# cpu和gpu之间传递数据，可通过copyto和as_in_context
y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())
# 两者的区别在于，copyto总是为目标变量创建新的内存，而如果源变量和目标变量的context一致，as_in_context会共享内存
# y.as_in_context(mx.gpu()) is y ----> True
# y.copyto(mx.gpu()) is y -----------> False



# GPU上的计算，数据都在GPU上，那么计算结果自动保存在相同的GPU上
# mxnet要求计算的所有输入数据都在同一个CPU或同一个GPU上

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())    # 注意，这里必须指定在GPU上，与y一致

v = net(y)
nd.save('v', v)



