# read input image using opencv
import numpy as np
import cv2
x_image = cv2.imread('test8.png', cv2.IMREAD_GRAYSCALE)
x_image = cv2.resize(x_image, dsize=(28, 28))

x_image = x_image / 255. #normalization
x_image = np.reshape(x_image, [-1, 28, 28, 1]) #according to the network input data size


# read from .pb file
import tensorflow as tf
with open('trained_model/frozen_model.pb', 'rb') as f:
    out_graph_def = tf.GraphDef()
    out_graph_def.ParseFromString(f.read())
    tf.import_graph_def(out_graph_def, name="")


    with tf.Session() as sess:
        data = sess.graph.get_tensor_by_name("data:0")
        prediction = sess.graph.get_tensor_by_name("prediction:0")

        sess.run(tf.global_variables_initializer())
        x_image_out = sess.run(prediction, feed_dict={data: x_image})

        print(np.argmax(x_image_out, 1))
