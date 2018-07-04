from mxnet import autograd, nd

# generate dataset
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.001, shape=labels.shape)

# read data
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X, y)
#     break

# define model
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))    # in gluon, fully connected layer is a Dense instance

# initial model params
from mxnet import init
net.initialize(init.Normal(sigma=0.01))    # weight initial as normal distribution
                                           # bias initial as all zero

# define loss function
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

# define optimize algorithm
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# train the model
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print('Epoch %d, loss %f' % (epoch, loss(net(features), labels).mean().asnumpy()))

# obatin the weight and bias
dense = net[0]
print(dense.weight.data())
print(dense.bias.data())
