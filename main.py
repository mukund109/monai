from utils import get_config, ImageHandler, write_image, read_image, postprocess, preprocess
from utils import maybe_make_directory
from model import CoarseFineModel
import tensorflow as tf
import cv2
import seaborn as sns
import numpy as np

def save_feature_histogram(sess, model, image_path, layer):
     image = cv2.resize(read_image(image_path), (model.input_shape[1], model.input_shape[0]))
     image = preprocess(image)
     feature_map = model.get_content_features(sess, image, [layer])[layer]
     feature_map = np.ndarray.flatten(feature_map)
     ax = sns.distplot(feature_map, kde=False, norm_hist=False)
     ax.set_title('Layer: {}, mean: {}'.format(layer, feature_map.mean()))
     ax.set_xlabel('activation_value')
     ax.set_ylabel('number of activations')
     maybe_make_directory('plots')
     ax.get_figure().savefig('plots/{}.png'.format(layer))

if __name__=='__main__':
    config = get_config()
    device = '/device:CPU:0' if config['debug'] else "/device:GPU:0"
    with tf.Session() as sess, tf.device(device):
        model = CoarseFineModel(sess, config, 50, 50)
        save_feature_histogram(sess, model, 'sample_images/aurat.jpeg', 'relu5_1')
    
