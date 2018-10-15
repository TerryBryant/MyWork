import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import loss as gloss, nn


# densenet与resnet的区别之一，就是densenet将相加改成了concat
# densenet由dense block（稠密层）和transition layer（过渡层）组成
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk


# dense block consists of several conv_blocks
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)  # nchw, concat on c
        return X


blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
print(Y.shape)  # 4, 23, 8, 8 --> 23 = 3 + 2*10


# 过渡层用于控制模型复杂度，否则一直堆叠下去，通道数太大
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


# blk = transition_block(10)
# blk.initialize()
# print(blk(Y).shape)

# densenet
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))


num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # 上一个稠密的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间加入过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add(transition_block(num_channels // 2))
net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(10))

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# # train
# lr, num_epochs, batch_size, ctx = 0.1, 5, 256, gb.try_gpu()
# net.initialize(ctx=ctx, init=init.Xavier())
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=96)
# gb.train_ch5(net, train_iter, test_iter, gloss.SoftmaxCrossEntropyLoss(), batch_size, trainer, ctx, num_epochs)


