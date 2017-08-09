import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os

input_folder = './runs/'  # './save_convnet'
input_ckpt =  './save/checkpoint'  # './save_convnet/convnet.ckpt'
output_graph = "./digit_classifier.pb"

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(input_folder)
input_checkpoint = checkpoint.model_checkpoint_path
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
checkpoint_file = tf.train.latest_checkpoint(input_folder)
with tf.Session() as sess:
    # saver.restore(sess, input_ckpt)
    saver.restore(sess, checkpoint_file)
    [print(n.name) for n in graph.as_graph_def().node]
    output_node = 'output_node'
    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        output_node.split(",")  # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

