from mxnet import nd
from mxnet.gluon import nn

# net = nn.Sequential()
# with net.name_scope():
# 	net.add(nn.Dense(256, activation="relu"))
# 	net.add(nn.Dense(10))
#
# print(net)

class MLP(nn.Block):
	def __init__(self, **kwargs):
		super(MLP, self).__init__(**kwargs)
		with self.name_scope():
			self.dense0 = nn.Dense(256)
			self.dense1 = nn.Dense(10)

	def forward(self, x):
		return self.dense1(nd.relu(self.dense0(x)))

net2 = MLP()
print(net2)

net2.initialize()
x = nd.random_uniform(shape=(4, 20))
y = net2(x)
<<<<<<< HEAD
print(y)
=======
print(y)
>>>>>>> origin/master
