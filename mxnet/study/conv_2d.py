from mxnet import autograd, nd
from mxnet.gluon import nn
import mxnet as mx


# 对单个元素的操作，使得无法自动求梯度
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), ctx=mx.gpu())
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w] * K).sum()

    return Y

# X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], ctx=mx.gpu())
# K = nd.array([[0, 1], [2, 3]], ctx=mx.gpu())
# Y = corr2d(X, K)

X = nd.ones((6, 8), ctx=mx.gpu())
X[:, 2:6] = 0
K = nd.array([[1, -1]], ctx=mx.gpu())
Y = corr2d(X, K)




class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


# 学习核数组
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize(ctx=mx.gpu())
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
    loss.backward()

    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, loss.sum().asscalar()))

print(conv2d.weight.data().reshape((1, 2)))