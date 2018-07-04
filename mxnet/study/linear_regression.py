from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.001, shape=labels.shape)

# plt.rcParams['figure.figsize'] = (3.5, 2.5)
# plt.scatter(features[:, 0].asnumpy(), labels.asnumpy(), 1)
# plt.show()


batch_size = 10

# data loader
def data_iter(batch_size, features, labels):
    num_examples = len(features)   # len returns the max of the dimensions
    indices = list(range(num_examples))

    # randomly read the samples
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break


# initial model params
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w, b]

# make gradient
for param in params:
    param.attach_grad()

# define model
def linreg(X, w, b):
    return nd.dot(X, w) + b

# define loss function
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# define optimize algorithm
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# train the model
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(1, num_epochs + 1):
    # suppose the size of samples can be divided by batch size
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()

        sgd([w, b], lr, batch_size)

    print('Epoch %d, loss %f' % (epoch, loss(net(features, w, b), labels).mean().asnumpy()))

print(w)
print(b)
