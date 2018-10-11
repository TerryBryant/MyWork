from mxnet import nd
from mxnet.gluon import nn

# x = nd.ones(3)
# nd.save('x', x)

# x2 = nd.load('x')
# print(x2)

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2, 20))
y = net(x)

filename = 'mlp.params'
net.save_parameters(filename)


# 实例化另一个mlp，用于对比参数
net2 = MLP()
net2.load_parameters(filename)
y2 = net2(x)

print(y2 == y)


# 通过load和save来读写NDArray
# 通过load_parameters和save_parameters来读写Gluon模型的参数