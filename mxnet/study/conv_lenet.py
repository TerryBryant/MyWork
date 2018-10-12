import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

# data
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)


# model
net = nn.Sequential()
# nn.Dense()会自动将n,c,h,w 转成n*c*h*w，这样就省去了flatten
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# # test model
# X = nd.random.uniform(shape=(1, 1, 28, 28))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)


# use gpu
def try_gpu4():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu4()

# evaluate
def evaluate_accuracy(data_iter, net, ctx):
    acc = nd.array([0], ctx=ctx)
    for X, y in data_iter:
        # 如果ctx是GPU，则将数据集复制到GPU上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        acc += gb.accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)

# train
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, start = 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_hat, y)

        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' %
              (epoch + 1, train_l_sum / len(train_iter), train_acc_sum / len(train_iter),
               test_acc, time.time() - start))

lr, num_epochs = 0.9, 5
net.initialize(init=init.Xavier(), force_reinit=True, ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)