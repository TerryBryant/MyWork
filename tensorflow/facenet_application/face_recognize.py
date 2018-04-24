import tensorflow as tf
import numpy as np
import cv2

image_size = 200
model_dir = "../trained_model/facenet_model/20180402-114759.pb"
image_name1 = "me1.JPG"
image_name2 = "me2.jpg"


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def image_preprocess(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = prewhiten(image)

    return image


with open(model_dir, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        scaled_reshape = []

        image1 = image_preprocess(image_name1)
        image2 = image_preprocess(image_name2)
        scaled_reshape.append(image1.reshape(-1, image_size, image_size, 3))
        scaled_reshape.append(image2.reshape(-1, image_size, image_size, 3))
        emb_array1 = np.zeros((1, embedding_size))
        emb_array2 = np.zeros((1, embedding_size))

        emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0],
                                                           phase_train_placeholder: False})[0]

        emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1],
                                                           phase_train_placeholder: False})[0]

        dist = np.sqrt(np.sum(np.square(emb_array1[0] - emb_array2[0])))
        print("128维特征向量的欧氏距离：%f " % dist)
