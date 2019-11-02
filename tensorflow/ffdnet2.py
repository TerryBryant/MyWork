import tensorflow as tf
import numpy as np  # can be removed, just used for generate fake train data
import tensorflow.contrib.slim as slim


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


class ffdnet(object):
    def __init__(self, sess, noise_sigma):
        self.model_name = 'ffdnet'
        self.sess = sess
        self.checkpoint_dir = './model'

        self.train_x = np.random.random((1, 512, 512, 3))
        self.train_y = np.random.random((1, 512, 512, 3))

        self.kernel_size = 3
        self.padding = 1
        self.num_feature_maps = 96
        self.num_conv_layers = 12
        self.input_features = 15
        self.output_features = 12
        self.idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.batch_num = 1
        self.noise_sigma = noise_sigma

    # define network
    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope('network', reuse=reuse):
            x = self.down_sample_image(x[0], self.noise_sigma)
            x = tf.expand_dims(x, 0)

            x = tf.layers.conv2d(inputs=x, filters=self.num_feature_maps, kernel_size=self.kernel_size,
                                 kernel_initializer=tf.initializers.variance_scaling,
                                 padding='SAME', use_bias=False,
                                 activation=tf.nn.relu)

            for _ in range(self.num_conv_layers - 2):
                x = tf.layers.conv2d(x, filters=self.num_feature_maps, kernel_size=self.kernel_size,
                                     kernel_initializer=tf.initializers.variance_scaling,
                                     padding='SAME', use_bias=False)
                x = tf.layers.batch_normalization(x, epsilon=1e-5, training=is_training)
                x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self.output_features,
                                 kernel_size=self.kernel_size, padding='SAME', use_bias=False)

            x = self.up_sample_image(x[0])
            x = tf.expand_dims(x, 0)

            return x

    def l2_loss(self, target, prediction):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.square(target - prediction))
        return loss

    def build_model(self):
        # graph input
        self.train_inputs = tf.placeholder(tf.float32, [1, 512, 512, 3], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [1, 512, 512, 3], name='train_labels')
        self.train_prediction = self.network(self.train_inputs)

        print(tf.trainable_variables())
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network'))

        self.train_loss = self.l2_loss(self.train_labels, self.train_prediction)

        # update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
        # with tf.control_dependencies(update_ops):
        #     train_step = tf.train.AdamOptimizer(1e-4).minimize(self.train_loss)

        self.optim = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(self.train_loss, var_list=tf.trainable_variables())

    def down_sample_image(self, inputs, noise_sigma):
        input_shape = inputs.shape

        Hout = input_shape[0] // 2
        Wout = input_shape[1] // 2
        Cout = input_shape[2] * 4

        output = tf.Variable(initial_value=lambda: tf.zeros(shape=[Hout, Wout, Cout], dtype=tf.float32), trainable=False)
        with tf.control_dependencies(
                output[:, :, idx:Cout:4].assign(inputs[self.idxL[idx][0]::2, self.idxL[idx][1]::2, :])
                for idx in range(4)):
            output = tf.identity(output)

        # add noise
        noisy_map = tf.fill([Hout, Wout, input_shape[2]], noise_sigma / 255.)
        output = tf.concat([noisy_map, output], axis=2)

        return output

    def up_sample_image(self, inputs):
        input_shape = inputs.shape

        out_h = input_shape[0] * 2
        out_w = input_shape[1] * 2
        out_c = input_shape[2] // 4
        shape_2 = input_shape[2]
        assert shape_2 % 2 == 0

        output = tf.Variable(initial_value=lambda: tf.zeros(shape=[out_h, out_w, out_c], dtype=tf.float32), trainable=False)
        with tf.control_dependencies(
                output[self.idxL[idx][0]::2, self.idxL[idx][1]::2, :].assign(inputs[:, :, idx:shape_2:4])
                for idx in range(4)):
            output = tf.identity(output)

        return output

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # print('-------------------------------')
        # vs = tf.trainable_variables()
        # print(vs)

        # saver to save model
        self.saver = tf.train.Saver()

        batch_x = self.train_x
        batch_y = self.train_y

        feed_dict = {
                    self.train_inputs: batch_x,
                    self.train_labels: batch_y,
                }

        _ = self.sess.run(self.optim, feed_dict=feed_dict)

        self.saver.save(self.checkpoint_dir, 1)


if __name__ == '__main__':
    with tf.Session() as sess:
        net = ffdnet(sess, noise_sigma=15.0)

        net.build_model()

        print('------------------------')
        show_all_variables()

        net.train()





