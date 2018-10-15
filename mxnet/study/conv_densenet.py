import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import nn

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


# 过渡层用于控制模型复杂度，否则一直对跌下去，通道数太大
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


