# inference using opencv
import numpy as np
import cv2

# read and convert the image to the format that similar to mnist image
x_image = cv2.imread('test8.png', cv2.IMREAD_GRAYSCALE)
x_image = cv2.resize(x_image, dsize=(28, 28))
x_image = 255 - x_image
x_image = x_image / 255.
x_image = x_image.astype(np.float32)

# read model and inference
inputBlob = cv2.dnn.blobFromImage(x_image)

net = cv2.dnn.readNetFromTensorflow('trained_model/frozen_model.pb')
net.setInput(inputBlob)
result = net.forward()

print(result)
print(np.argmax(result, 1))


################################################################################################
# inference using tensorflow
import tensorflow as tf
import numpy as np
import cv2

# read and convert the image to the format that similar to mnist image
x_image = cv2.imread('test8.png', cv2.IMREAD_GRAYSCALE)
x_image = cv2.resize(x_image, dsize=(28, 28))
x_image = 255 - x_image
x_image = x_image / 255.
x_image = x_image.astype(np.float32)


# read model and inference
inputBlob = np.reshape(x_image, [-1, 28, 28, 1])

with open('trained_model/frozen_model.pb', 'rb') as f:
    out_graph_def = tf.GraphDef()
    out_graph_def.ParseFromString(f.read())
    tf.import_graph_def(out_graph_def, name="")


    with tf.Session() as sess:
        data = sess.graph.get_tensor_by_name("input/x_input:0")
        prediction = sess.graph.get_tensor_by_name("softmax/prediction:0")

        sess.run(tf.global_variables_initializer())
        x_image_out = sess.run(prediction, feed_dict={data: inputBlob})

        print(x_image_out)
        print(np.argmax(x_image_out, 1))
