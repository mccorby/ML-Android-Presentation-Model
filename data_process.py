import argparse

import sys
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    batch = mnist.train.next_batch(12)
    print(len(batch))
    print(len(batch[0][0]))

    fig = plt.figure(1, (4., 4.))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i in range(12):
        grid[i].imshow(_create_image(batch, i), cmap='gray')

    plt.show()


def _create_image(batch, index):
    image = np.array(batch[0][index])
    return image.reshape((28, 28))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
