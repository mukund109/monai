from vgg19 import build_vgg
import tensorflow as tf
import numpy as np
from logger import CustomLogger
from utils import postprocess

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
    def __init__(self, sess, config, height, width):
        self.config = config
        self._build_network(sess, height, width, 3)
        self.logger = None

    def _build_style_loss(self):
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
                diff =  self.style_gram[layer] - self.input_gram[layer]
                loss = tf.reduce_sum(tf.pow((diff), 2)) / (4 * num_features**2)
                style_losses.append(loss * weight)

        self.style_loss = tf.add_n(style_losses, name='cumulative_style_loss')

    def _build_content_loss(self):
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

    def _build_network(self, sess, h, w, d):
        self.net = build_vgg(sess, h, w, d, self.config)
        self._build_style_loss()
        self._build_content_loss()

        with tf.name_scope('total_variation_loss'):
            self.tv_loss = tf.reduce_sum(tf.image.total_variation(self.net['input']))

        # TODO: denoising loss, weighted sum
        with tf.name_scope('total_loss'):
            style_wt = self.config['style_weight']
            content_wt = self.config['content_weight']
            tv_weight = self.config['tv_weight']
            self.total_loss = style_wt*self.style_loss + content_wt*self.content_loss + tv_weight*self.tv_loss

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

    def _get_feed_dict(self, sess, content_image, style_images, weights):
        "Only to be used when stylizing a single image, ie when using 'stylize()' "
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

    def _neural_matching(self, sess, tiles, style_images):
        layer = self.config['neural_matching_layer']
        if self.config['verbose']: print('Computing neural matches')
        style_tiles = []
        for image in style_images:
            style_tiles += image

        tile_features = []
        for tile in tiles:
            sess.run(self.net['input'].assign(tile))
            features = np.ndarray.flatten(sess.run(self.net[layer]))
            features = features/np.linalg.norm(features)
            tile_features.append(features)
        tile_features = np.stack(tile_features)

        style_features = []
        for style in style_tiles:
            sess.run(self.net['input'].assign(style))
            features = np.ndarray.flatten(sess.run(self.net[layer]))
            features = features/np.linalg.norm(features)
            style_features.append(features)
        style_features = np.stack(style_features)

        similarity_matrix = np.matmul(tile_features, style_features.T)
        if self.config['verbose']:print('Computed {}x{} similarity matrix'.format(*similarity_matrix.shape))

        mapping = np.argmin(similarity_matrix, axis=1)
        matches = []
        for index in mapping:
            matches.append(style_tiles[index])

        return matches


    def stylize_multiple(self, sess, tiles, style_images, init_tiles, update_fn=None):
        style_weights = self.config['style_image_weights']

        with tf.device('/device:cpu:0'):
            logger = self.init_logger(sess)

        # initializing optimizer
        if self.config['verbose']: print("Initializing optimizer")
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B',
                options={'maxiter': self.config['max_iterations_per_tile'],
                          'disp': 50})

        for num_pass in range(self.config['max_image_passes']):
            for i, tile in enumerate(tiles):

                #Logging important stuff
                with tf.device('/device:cpu:0'):
                    logger.log_images("Content_image", [postprocess(tile)])
                    # TODO: log all style tiles
                    logger.log_images("Style_images", list(map(postprocess, style_images[0])))
                    logger.log_images("Init_image", [postprocess(init_tiles[i])])


                if self.config['verbose']: print('COMPUTING FEED FOR TILE {}'.format(i))
                feed_dict = dict()
                # Getting the content features of the image tile , and appending them
                # to the feed dict
                content_features = self.get_content_features(sess, tile)
                for layer, placeholder in self.content_features.items():
                    feed_dict[placeholder] = content_features[layer]

                # Getting the style features of the style tiles, and appending them
                # to the feed dict
                for style_tiles, weight in zip(style_images, style_weights):
                    for style_tile in style_tiles:
                        grams = self.get_style_gram(sess, style_tile)
                        for layer, placeholder in self.style_gram.items():
                            if placeholder not in feed_dict.keys():
                                feed_dict[placeholder] = weight*grams[layer]
                            else:
                                feed_dict[placeholder] += weight*grams[layer]

                # The gram matrices of the init tiles that are NOT being
                # optimized are subtracted from the overall style gram matrix
                if not self.config['optimize_tile_with_global_gram']:
                    o_tiles = init_tiles[:i] + init_tiles[i+1:]
                    for o_tile in o_tiles:
                        grams = self.get_style_gram(sess, o_tile)
                        for layer, placeholder in self.style_gram.items():
                            feed_dict[placeholder] -= grams[layer]

                output = self.run_optimizer(sess, optimizer, logger, feed_dict, init_tiles[i])

                # Update the initialization tiles
                if update_fn is None:
                    init_tiles[i] = output
                else:
                    init_tiles = update_fn(output, i, init_tiles)

        return init_tiles

    def stylize_patches(self, sess, tiles, style_images, init_tiles, update_fn):

        #get a style tile for every content tile
        matches = self._neural_matching(sess, tiles, style_images)

        with tf.device('/device:cpu:0'):
            logger = self.init_logger(sess)

        # initializing optimizer
        if self.config['verbose']: print("Initializing optimizer")
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B',
                options={'maxiter': self.config['max_iterations_per_tile'],
                          'disp': 50})

        for num_pass in range(self.config['max_image_passes']):
            for i, tile in enumerate(tiles):
                feed_dict = self._get_feed_dict(sess, tile, [matches[i]], [1])

                #Logging important stuff
                with tf.device('/device:cpu:0'):
                    logger.log_images("Content_image", [postprocess(tile)])
                    # TODO: log all style tiles
                    logger.log_images("Style_images", [postprocess(matches[i])])
                    logger.log_images("Init_image", [postprocess(init_tiles[i])])

                output = self.run_optimizer(sess, optimizer, logger, feed_dict, init_tiles[i])

                #update initialization tiles
                if update_fn is None:
                    init_tiles[i] = output
                else:
                    init_tiles = update_fn(output, i, init_tiles)

        return init_tiles

    def stylize(self, sess, content_image, style_images, init_img):
        feed_dict = self._get_feed_dict(sess, content_image, style_images, \
                                            self.config['style_image_weights'])

        #Logging important stuff
        with tf.device('/device:cpu:0'):
            logger = self.init_logger(sess)
            logger.log_images("Content_image", [postprocess(content_image)])
            logger.log_images("Style_images", list(map(postprocess, style_images)))
            logger.log_images("Init_image", [postprocess(init_img)])

        # initializing optimizer
        if self.config['verbose']: print("Initializing optimizer")
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B',
                options={'maxiter': self.config['max_iterations'],
                          'disp': 50})

        output = self.run_optimizer(sess, optimizer, logger, feed_dict, init_img)

        return output

    def run_optimizer(self, sess, optimizer, logger, feed_dict, init_img, log_iter=50):
        if self.config['verbose']: print("Running optimizer")

        # This is a callback function that writes summary
        def _step_summary(tloss, sloss, closs, tvloss):
            with tf.device('/device:cpu:0'):
                #increment global step
                if logger.global_step%5 == 0:
                    logger.log_scalar("Total_loss", tloss)
                    logger.log_scalar("Style_loss", sloss)
                    logger.log_scalar("Content_loss", closs)
                    logger.log_scalar("Total_variation", tvloss)

        # A callback function that increments global step
        def _step_callback(image):
            logger.increment_global_step()
            if logger.global_step%log_iter == 0:
                image = np.reshape(image, self.net['input'].get_shape().as_list())
                logger.log_images("Stylized_image", [postprocess(image)])

        sess.run(self.net['input'].assign(init_img))
        optimizer.minimize(sess, feed_dict, fetches=[self.total_loss, \
                            self.style_loss, self.content_loss,\
                            self.tv_loss], \
                            step_callback= _step_callback, \
                            loss_callback= _step_summary)
        return sess.run(self.net['input'])

    def init_logger(self, sess):
        if self.logger is None:
            self.logger = CustomLogger(self.config['log_dir'], sess.graph)

        return self.logger

if __name__=='__main__':
    from utils import get_config, get_content_image, get_style_images, get_init_image
    import matplotlib.pyplot as plt
    config = get_config()
    a = CoarseFineModel(config)
    output = a.run(40,40,3, content_img, style_imgs, init_img)
    plt.imsave('output.jpg', output[0])
