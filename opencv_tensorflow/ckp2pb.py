import os
import tensorflow as tf

########################################################################
# .ckp file to .pb file(without weight parameters)
checkpoint = tf.train.get_checkpoint_state('trained_model/')
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = 'trained_model/' + "frozen_model2.pb"
output_node_names = 'softmax/prediction'
# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True



# We import the meta graph in the current default Graph
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

# We retrieve the protobuf graph definition
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

# We start a session and restore the graph weights
with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)
    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))



##########################################################################################
# # .ckp file to .pb and .pbtxt file
# dir_path = 'trained_model/'
# save_dir = 'trained_model/Protobufs'
# sess = tf.Session()
# print("Restoring the model to the default graph ...")
# saver = tf.train.import_meta_graph(dir_path + '/mnist_cnn.ckpt.meta')
# saver.restore(sess,tf.train.latest_checkpoint(dir_path))
# print("Restoring Done .. ")
#
# print("Saving the model to Protobuf format: ", save_dir)
#
# #Save the model to protobuf  (pb and pbtxt) file.
# tf.train.write_graph(sess.graph_def, save_dir, "Binary_Protobuf.pb", False)
# tf.train.write_graph(sess.graph_def, save_dir, "Text_Protobuf.pbtxt", True)
# print("Saving Done .. ")