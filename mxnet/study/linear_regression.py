import mxnet.ndarray as nd
import mxnet.autograd as ag

# 创建数据集
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

print(X[0], y[0])

import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
plt.show()

import random
batch_size = 10

def data_iter():
	# 产生一个随机索引
	idx = list(range(num_examples))
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		j = nd.array(idx[i:min(i+batch_size, num_examples)])
		yield nd.take(X, j), nd.take(y, j)

for data, label in data_iter():
	print(data, label)
	break
