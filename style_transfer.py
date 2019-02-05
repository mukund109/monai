import tensorflow as tf
import numpy as np
from model import build_model
from utils import get_content_image, get_style_images
import struct
import errno
import time
import cv2
import os
import json

config = json.load(open('config.json'))

def content_layer_loss(p, x):
    """
    Calculates the content layer loss between content tensor p, and
    tensor variable x
    """
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if config['content_loss_function']   == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif config['content_loss_function'] == 2:
        K = 1. / (N * M)
    elif config['content_loss_function'] == 3:
        K = 1. / 2.
    else:
        raise ValueError()

    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def style_layer_loss(a, x):
    """
    Calculates style layer loss between style image representations a and
    tensor variable x
    """
    with tf.name_scope("style_layer_loss"):
        _, h, w, d = a.get_shape()
        M = h.value * w.value
        N = d.value
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, num_activations, num_features):
    """
    Returns gram matrix, given matrix X, with shape (num_activations, num_features)
    """
    F = tf.reshape(x, (num_activations, num_features))
    G = tf.matmul(tf.transpose(F), F)
    return G

def sum_style_losses(sess, net, style_imgs):
    """
    Computes the feature activations of the style images on the layers
    specified by config['style_layers'], then defines the tensorflow op that is
    supposed to calculate the style loss using these activations
    """
    total_style_loss = 0.
    weights = config['style_imgs_weights']
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(config['style_layers'], config['style_layer_weights']):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a, name="style_features_{}".format(layer))
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(config['style_layers']))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_content_losses(sess, net, content_img):
    """
    Computes the feature activations of the content image on the layers
    specified by config['content_layers'], then defines the tensorflow op that is
    supposed to calculate the style loss using these activations
    """
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(config['content_layers'], config['content_layer_weights']):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(config['content_layers']))
    return content_loss

def get_optimizer(loss):
    if config['optimizer'] == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method='L-BFGS-B',
                options={'maxiter': config['max_iterations'],
                          'disp': 50})
    elif config['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(config['learning_rate'])
    return optimizer

def minimize_with_lbfgs(sess, net, optimizer, init_img):
    if config['verbose']: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss, content_img):
    if config['verbose']: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < config['max_iterations']):
        sess.run(train_op)
        if config['verbose'] and (iterations%50 == 0):
            curr_loss = loss.eval()

            #Save checkpoint
            checkpoint_img = sess.run(net['input'])

            # TODO: save checkpoint in original colors?
            # if config['original_colors']:
            #     checkpoint_img = convert_to_original_colors(content_img, checkpoint_img)
            save_checkpoint(checkpoint_img, iterations)

            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


def stylize(content_img, style_imgs, init_img, frame=None):
    if config['debug']:
        config['device']="/cpu:0"
    with tf.device(config['device']), tf.Session() as sess:
        # setup network
        net = build_model(content_img)

        # style loss
        L_style = sum_style_losses(sess, net, style_imgs)

        # content loss
        L_content = sum_content_losses(sess, net, content_img)

        # denoising loss
        L_tv = tf.image.total_variation(net['input'])

        # loss weights
        alpha = config['content_weight']
        beta  = config['style_weight']
        theta = config['tv_weight']

        # total loss
        L_total  = alpha * L_content
        L_total += beta  * L_style
        L_total += theta * L_tv

    # optimization algorithm
    optimizer = get_optimizer(L_total)

    if config['optimizer'] == 'adam':
        minimize_with_adam(sess, net, optimizer, init_img, L_total, np.copy(content_img))
    elif config['optimizer'] == 'lbfgs':
        minimize_with_lbfgs(sess, net, optimizer, init_img)

    output_img = sess.run(net['input'])

    # TODO: CHANGE THIS
    # if config['original_colors']:
    #     output_img = convert_to_original_colors(np.copy(content_img), output_img)

    write_image_output(output_img, content_img, style_imgs, init_img)

def render_single_image():
    content_img = get_content_image(config['content_img'])
    style_imgs = get_style_images(content_img)
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = get_init_image(config['init_img_type'], content_img, \
                            style_imgs, init_img_path = config['init_img_path'])
        tick = time.time()
        stylize(content_img, style_imgs, init_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))
