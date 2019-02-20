from utils import get_config
print("what what what what")
import tensorflow as tf
import numpy as np
import scipy.io


def build_vgg(sess, h, w, d, config):

    def conv_layer(layer_name, layer_input, W):
        padding = tf.constant([[0,0],[1,1],[1,1],[0,0]], name='padding')
        layer_input = tf.pad(layer_input, padding, mode='SYMMETRIC', name='padding_op')
        conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='VALID')
        if config['verbose']: print('--{} | shape={} | weights_shape={}'.format(layer_name,
            conv.get_shape(), W.get_shape()))
        return conv

    def relu_layer(layer_name, layer_input, b):
        relu = tf.nn.relu(layer_input + b)
        if config['verbose']:
            print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(),
              b.get_shape()))
        return relu

    def pool_layer(layer_name, layer_input):
        if config['pooling_type'] == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
        elif config['pooling_type'] == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
        if config['verbose']:
            print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
        return pool

    # helper function for loading VGG weights
    def get_weights(vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]

        #This way of initialization is a workaround to ensure that the weights
        #are not written to the event file
        # https://stackoverflow.com/questions/35903910/tensorflow-reading-the-entire-contents-of-a-file-exactly-once-into-a-single-ten/35904439#35904439
        weight_init = tf.placeholder(tf.float32, shape=weights.shape)
        W = tf.Variable(weight_init, trainable=False)
        sess.run(W.initializer, feed_dict={weight_init:weights})
        return W

    #helper function for loading VGG bias weights
    def get_bias(vgg_layers, i):
        bias = vgg_layers[i][0][0][2][0][1]
        bias = np.reshape(bias, (bias.size))
        bias_init = tf.placeholder(tf.float32, shape=bias.shape)
        b = tf.Variable(bias_init, trainable=False)
        sess.run(b.initializer, feed_dict={bias_init:bias})
        return b


    if config['verbose']: print('\nBUILDING VGG-19 NETWORK')
    net = {}

    if config['verbose']: print('loading model weights...')
    vgg_rawnet     = scipy.io.loadmat(config['model_weights'])
    vgg_layers     = vgg_rawnet['layers'][0]
    if config['verbose']: print('constructing layers...')
    net['input']   = tf.Variable(tf.zeros((1, h, w, d), dtype=np.float32), name='input')

    if config['verbose']: print('LAYER GROUP 1')
    with tf.name_scope('layer1'):
        net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
        net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

        net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
        net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))

        net['pool1']   = pool_layer('pool1', net['relu1_2'])

    if config['verbose']: print('LAYER GROUP 2')
    with tf.name_scope('layer2'):
        net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
        net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))

        net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
        net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))

        net['pool2']   = pool_layer('pool2', net['relu2_2'])

    if config['verbose']: print('LAYER GROUP 3')
    with tf.name_scope('layer3'):
        net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
        net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

        net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
        net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

        net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
        net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

        net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
        net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

        net['pool3']   = pool_layer('pool3', net['relu3_4'])

    if config['verbose']: print('LAYER GROUP 4')
    with tf.name_scope('layer4'):
        net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
        net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

        net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
        net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

        net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
        net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

        net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
        net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

        net['pool4']   = pool_layer('pool4', net['relu4_4'])

    if config['verbose']: print('LAYER GROUP 5')
    with tf.name_scope('layer5'):
        net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
        net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

        net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
        net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

        net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
        net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

        net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
        net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

        net['pool5']   = pool_layer('pool5', net['relu5_4'])

    return net
