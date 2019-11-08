import tensorflow as tf
import numpy as np


idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]


# def down_sample_image(inputs, noise_sigma):
#     input_shape = tf.shape(inputs)
#     output0 = inputs[idxL[0][0]::2, idxL[0][1]::2, :]
#     output1 = inputs[idxL[1][0]::2, idxL[1][1]::2, :]
#     output2 = inputs[idxL[2][0]::2, idxL[2][1]::2, :]
#     output3 = inputs[idxL[3][0]::2, idxL[3][1]::2, :]
#
#     output_list = []
#     for i in range(3):
#         output_list.append(tf.stack([output0[:, :, i], output1[:, :, i], output2[:, :, i], output3[:, :, i]], axis=2))
#     output = tf.concat(output_list, axis=2)
#
#     # add noise
#     noisy_map = tf.fill([input_shape[0] // 2, input_shape[1] // 2, input_shape[2]], noise_sigma / 255.)
#     output = tf.concat([noisy_map, output], axis=2)
#
#     return output


def down_sample_image(inputs, noise_sigma):
    input_shape = inputs.shape

    Hout = input_shape[0].value // 2
    Wout = input_shape[1].value // 2
    Cout = input_shape[2].value * 4

    output = tf.Variable(initial_value=lambda: tf.zeros(shape=[Hout, Wout, Cout], dtype=tf.float32), trainable=False)
    with tf.control_dependencies(
            output[:, :, idx:Cout:4].assign(inputs[idxL[idx][0]::2, idxL[idx][1]::2, :])
            for idx in range(4)):
        output = tf.identity(output)

    # add noise
    noisy_map = tf.fill([Hout, Wout, input_shape[2].value], noise_sigma/255.)
    output = tf.concat([noisy_map, output], axis=2)

    return output


def down_sample_image2(inputs, noise_sigma):
    output = tf.nn.space_to_depth(inputs, 2)
    output0 = output[:, :, :, ::3]
    output1 = output[:, :, :, 1::3]
    output2 = output[:, :, :, 2::3]
    output = tf.concat([output0, output1, output2], axis=3)

    # add noise
    noisy_map = tf.fill([output.shape[0].value, output.shape[1].value, output.shape[2].value, inputs.shape[3].value], noise_sigma / 255.)
    output = tf.concat([noisy_map, output], axis=3)

    return output


def up_sample_image(inputs):
    input_shape = inputs.shape

    Hout = input_shape[0].value
    Wout = input_shape[1].value
    Cout = input_shape[2].value
    output = tf.Variable(initial_value=lambda: tf.zeros(shape=[Hout, Wout, Cout], dtype=tf.float32), trainable=False)
    with tf.control_dependencies(tf.assign(ref=output, value=inputs, use_locking=True)):
        output = tf.identity(output)

    return output



# def up_sample_image(inputs):
#     input_shape = inputs.shape
#
#     Hout = input_shape[0].value * 2
#     Wout = input_shape[1].value * 2
#     Cout = input_shape[2].value // 4
#     shape_2 = input_shape[2].value
#
#     assert shape_2 % 2 == 0
#
#     output = tf.Variable(initial_value=lambda: tf.zeros(shape=[Hout, Wout, Cout], dtype=tf.float32), trainable=False)
#     with tf.control_dependencies(
#             tf.assign(ref=output[idxL[idx][0]::2, idxL[idx][1]::2, :], value=inputs[:, :, idx:shape_2:4], use_locking=True)
#             for idx in range(4)):
#         output = tf.identity(output)
#
#     return output


# tf.nn.depth_to_space
def up_sample_image2(inputs):
    output0 = inputs[:, :, :, ::4]
    output1 = inputs[:, :, :, 1::4]
    output2 = inputs[:, :, :, 2::4]
    output3 = inputs[:, :, :, 3::4]
    output = tf.concat([output0, output1, output2, output3], axis=3)
    output = tf.nn.depth_to_space(output, 2)

    return output


if __name__ == '__main__':
    # upsample
    input = np.arange(96).reshape((3, 2, 2, 8))
    # input = np.expand_dims(input, 0)
    print(input)

    a = tf.placeholder(dtype=tf.float32, shape=[None, 2, 2, 8])

    output = tf.map_fn(lambda x: up_sample_image(x), a, dtype=tf.float32, parallel_iterations=10)
    # output = up_sample_image2(a)




    # # down sample
    # input = np.arange(192).reshape((1, 8, 8, 3))
    # print(input)
    # a = tf.placeholder(dtype=tf.float32, shape=[1, 8, 8, 3])
    # output = tf.map_fn(lambda x: down_sample_image(x, 255), a, dtype=tf.float32, parallel_iterations=1)
    # # output = down_sample_image2(a, 255)






    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hh = sess.run(output, feed_dict={a: input})

        print(hh)
        print(hh.shape)