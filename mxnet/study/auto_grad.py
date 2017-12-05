import mxnet.autograd as ag
import mxnet.ndarray as nd


x = nd.array([[1, 2], [3, 4]])
x.attach_grad()

with ag.record():
    y = x * x
    z = y * x * x

z.backward()
print(x.grad)

##########################################################

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with ag.record():
    y = x * x
    z = y * x * x

head_gradient1 = nd.array([[1, 1], [1, 1]])
z.backward(head_gradient1)
print(x.grad)

##########################################################

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with ag.record():
    y = x * x
    z = y * x * x

head_gradient2 = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient2)
print(x.grad)

##########################################################
# output
[[   4.   32.]
 [ 108.  256.]]
<NDArray 2x2 @cpu(0)>

[[   4.   32.]
 [ 108.  256.]]
<NDArray 2x2 @cpu(0)>

[[ 40.          32.        ]
 [ 10.80000019   2.55999994]]
<NDArray 2x2 @cpu(0)>

# 总结：所谓的头梯度就是系数，不加头梯度，默认系数为1