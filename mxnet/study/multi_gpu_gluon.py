import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
import time


def resnet18(num_classes):
    def res_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(gb.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(gb.Residual(num_channels))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(res_block(64, 2, first_block=True),
            res_block(128, 2),
            res_block(256, 2),
            res_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


net = resnet18(10)
# X = nd.random.normal(shape=(1, 1, 224, 224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'outputshape:\t', X.shape)


# ctx = gb.try_gpu()
# net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
# X = nd.random.normal(shape=(4, 1, 28, 28))
# gpu_x = gutils.split_and_load(X, ctx)
# print(net(gpu_x[0]))
#
#
# # 注意默认下weight.data()会返回CPU上的参数值
# weight = net[0].params.get('weight')
# try:
#     weight.data()
# except RuntimeError:
#     print('not initialized on', mx.cpu())
# print(weight.data(ctx[0])[0])


# 多GPU训练模型
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y) for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, training time: %.1f sec, test_acc %.2f' % (
            epoch + 1, train_time, test_acc))


train(num_gpus=1, batch_size=256, lr=0.1)