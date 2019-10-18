# import cv2
import numpy as np
import tensorflow as tf


# # input = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
# a = tf.placeholder(dtype=tf.int32, shape=[1, 4, 4])
#
# idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
# # output = tf.Variable(initial_value=tf.zeros(shape=[Cout, Hout, Wout]))
#
# output_list = []
# for idx in range(4):
#     output_list.append(a[:, idxL[idx][0]::2, idxL[idx][1]::2])
# output = tf.concat(output_list, axis=0)


# indices = [[0, 0], [1, 1]]
# params = [['a', 'b'], ['c', 'd']]
# k = tf.gather_nd(params, indices)





input = np.array([[[ 1,  3],[ 9, 11]], [[ 2,  4], [10, 12]], [[ 5,  7], [13, 15]], [[ 6,  8], [14, 16]]])

a = tf.placeholder(dtype=tf.int32, shape=[4, 2, 2])


a_shape = a.shape
sca = 2
sca2 = sca*sca
Cout = a_shape[0].value//sca2
Hout = a_shape[1].value*sca
Wout = a_shape[2].value*sca
idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

assert a_shape[0].value % 2 == 0


output = tf.Variable(initial_value=tf.zeros(shape=[Cout, Hout, Wout], dtype=tf.int32))

# output1 = tf.concat([a[0:4:4, :, :], a[1:4:4, :, :]], axis=2)
# output2 = tf.concat([a[2:4:4, :, :], a[3:4:4, :, :]], axis=2)
# output = tf.concat([output1, output2], axis=1)


# for idx in range(4):
#     tf.assign(output[:, idxL[idx][0]::sca, idxL[idx][1]::sca], a[idx:a_shape[0].value:sca2, :, :])

output_tmp = tf.scatter_update(output, [0, 0, 0], a[0:4:4, :, :])



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hh = sess.run(output_tmp, feed_dict={a: input})
    print(hh)
    print(hh.shape)


    # ans = sess.run(output, feed_dict={a: input})
    # print(ans)
    # print(ans.shape)


# img = cv2.imread('D:\\Workspace\\matlab\\CBDNet-master\\testsets\\an\\1.jpg')
# img = cv2.resize(img, (500, 400))

# img = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# img2 = np.expand_dims(img, 0)

# img3 = np.concatenate((img2, img2[:, :, -1, :][:, :, np.newaxis, :]), axis=3)

# print(img)
# print(img2.shape)
