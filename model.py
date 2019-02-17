from vgg19 import build_vgg
import tensorflow as tf
import numpy as np
from logger import CustomLogger

def _gram_matrix(x, num_activations, num_features):
    """
    Returns gram matrix, given matrix X, with shape (num_activations, num_features)
    """
    F = tf.reshape(x, (-1, num_activations, num_features))
    G = tf.matmul(tf.transpose(F, perm=[0,2,1]), F)

    # this takes the sum of the gram matrices of all images in the batch
    G = tf.reduce_sum(G, axis=0)

    G = G/num_activations
    return G

class CoarseFineModel(object):
    def __init__(self, config):
        self.config = config

    def build_style_loss(self):
        """
        This function constructs the tensorfow ops required to calculate the
        style loss. The gram matrices of the style image are declared as
        placeholders
        """
        self.style_gram = dict()
        self.input_gram = dict()
        style_losses = []
        self.style_loss = 0.
        for layer, weight in zip(self.config['style_layers'], self.config['style_layer_weights']):

            x = self.net[layer]
            _, feature_h, feature_w, num_features = x.get_shape()
            feature_size = feature_h.value * feature_w.value
            num_features = num_features.value

            self.style_gram[layer] = tf.placeholder(dtype=tf.float32, \
                                        shape=[num_features]*2, \
                                        name = "style_gram_{}".format(layer))

            with tf.name_scope("input_gram_{}".format(layer)):
                self.input_gram[layer] = _gram_matrix(x, feature_size, num_features)

            with tf.name_scope("style_loss_{}".format(layer)):
                diff = self.input_gram[layer] - self.style_gram[layer]
                loss = tf.reduce_sum(tf.pow((diff), 2)) / (4 * num_features**2)
                style_losses.append(loss * weight)

        self.style_loss = tf.add_n(style_losses, name='cumulative_style_loss')

    def build_content_loss(self):
        """
        This function constructs the tensorfow ops required to calculate the
        content loss. The feature maps of the content image are declared as
        placeholders
        """
        self.content_features = dict()
        self.content_loss = 0.
        for layer, weight in zip(self.config['content_layers'], \
                                        self.config['content_layer_weights']):
            x = self.net[layer]
            b, h, w, d = x.get_shape()
            b, h, w, d = b.value, h.value, w.value, d.value
            feature_size, num_features = h*w, d

            y = tf.placeholder(dtype=tf.float32, \
                                    shape=(b, h, w, d), \
                                    name = "content_features_{}".format(layer))

            self.content_features[layer] = y

            with tf.name_scope('content_loss'):
                K = num_features*feature_size
                norm = tf.convert_to_tensor(1/K, name='normalization_constant')
                loss = norm * tf.reduce_sum(tf.pow((x - y), 2))
                self.content_loss += loss * weight

    def build_network(self, sess, h, w, d):
        self.net = build_vgg(sess, h, w, d, self.config)
        self.build_style_loss()
        self.build_content_loss()

        # TODO: denoising loss, weighted sum
        with tf.name_scope('total_loss'):
            style_wt = self.config['style_weight']
            content_wt = self.config['content_weight']
            self.total_loss = style_wt*self.style_loss + content_wt*self.content_loss

    def get_content_features(self, sess, image):
        """
        Computes the features maps of the image, corresponding to the layers
        mentioned in config['content_layers'], returns a dict of numpy arrays
        """
        assert image.shape == self.net['input'].shape
        sess.run(self.net['input'].assign(image))
        feature_maps = dict()
        for layer in self.config['content_layers']:
            feature_maps[layer] = sess.run(self.net[layer])
        return feature_maps

    def get_style_gram(self, sess, image):
        """
        Computes the gram matrices of the image, corresponding to the features
        mentioned in config['style_layers'], returns a dict of numpy arrays
        """
        assert image.shape == self.net['input'].shape
        sess.run(self.net['input'].assign(image))
        gram_matrices = dict()
        for layer in self.config['style_layers']:
            gram_matrices[layer] = sess.run(self.input_gram[layer])

        return gram_matrices

    def get_feed_dict(self, sess, content_image, style_images, weights):
        feed_dict = dict()
        #Feeding content features
        content_features = self.get_content_features(sess, content_image)
        for layer, placeholder in self.content_features.items():
            feed_dict[placeholder] = content_features[layer]

        #Feeding style features
        for image, weight in zip(style_images, weights):
            grams = self.get_style_gram(sess, image)
            for layer, placeholder in self.style_gram.items():
                if placeholder not in feed_dict.keys():
                    feed_dict[placeholder] = weight*grams[layer]
                else:
                    feed_dict[placeholder] += weight*grams[layer]

        return feed_dict

    def stylize(self, sess, content_image, style_images, init_img):
        feed_dict = self.get_feed_dict(sess, content_image, style_images, \
                                            self.config['style_image_weights'])

        logger = self.init_logger(sess)

        # initializing optimizer
        if self.config['verbose']: print("Initializing optimizer")
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B',
                options={'maxiter': self.config['max_iterations'],
                          'disp': 50})

        output = self.run_optimizer(sess, optimizer, logger, feed_dict, init_img)

        return output

    def run_optimizer(self, sess, optimizer, logger, feed_dict, init_img, num_iters=50):

        # This is a callback function that increments global step and writes summary
        # its a workaround because scipy optimizers don't update the graph
        # at every step, so instead 'step_summary' is passed as a callback to the
        # 'optimizer', and custom summary statistics are recorded
        # https://stackoverflow.com/questions/44685228/how-to-get-loss-function-history-using-tf-contrib-opt-scipyoptimizerinterface
        def _step_summary(tloss, sloss, closs, image):
            #increment global step
            logger.increment_global_step()
            step_value = logger.global_step
            if step_value%5 == 0:
                logger.log_scalar("Total_loss", tloss, step_value)
                logger.log_scalar("Style_loss", sloss, step_value)
                logger.log_scalar("Content_loss", closs, step_value)
            if step_value%50 == 0:
                # TODO: postprocess image before writing
                logger.log_images("Stylized_image", [image[0]], step_value)

        sess.run(self.net['input'].assign(init_img))
        optimizer.minimize(sess, feed_dict, fetches=[self.total_loss, \
                            self.style_loss, self.content_loss,\
                            self.net['input']], loss_callback= _step_summary)
        return sess.run(self.net['input'])

    def init_logger(self, sess, custom=True):
        if custom:
            return CustomLogger(self.config['log_dir'], sess.graph)

    def run(self, h, w, d, content, styles, init):
        with tf.Session() as sess:
            self.build_network(sess, h, w, d)
            output = self.stylize(sess, content, styles, init)
        return output

if __name__=='__main__':
    from utils import get_config, get_content_image, get_style_images, get_init_image
    import matplotlib.pyplot as plt
    config = get_config()
    content_img = get_content_image(config['content_image'], (40,40))
    style_imgs = get_style_images(config['style_images'], (40,40))
    init_img = get_init_image(config['init_image_type'], content_img, \
                        style_imgs, init_img_path = config['init_image_path'])
    a = CoarseFineModel(config)
    output = a.run(40,40,3, content_img, style_imgs, init_img)
    plt.imsave('output.jpg', output[0])
