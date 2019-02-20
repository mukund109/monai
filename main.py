from utils import get_config, ImageHandler, write_image, postprocess
from model import CoarseFineModel
import tensorflow as tf

if __name__=='__main__':
    config = get_config()

    handler = ImageHandler(config, base_size=(400,400))
    content, styles, init = handler.get_coarse_data()

    device = '/device:CPU:0' if config['debug'] else "/device:GPU:0"
    with tf.Session() as sess, tf.device(device):
        model = CoarseFineModel(sess, config, 400, 400)
        output = model.stylize(sess, content, styles, init)
        init = postprocess(output)
        tiles, style_tiles, init_tiles = handler.get_overlapping_data(zoom=4, overlap=4,init_image=init)
        update_fn = handler.get_update_function(overlap=4)
        output = model.stylize_patches(sess, tiles, style_tiles, init_tiles, update_fn)
    handler.write_overlapping_image_output(output, tiles, style_tiles, init_tiles, overlap=4)


    if False:
        content, styles, init = handler.get_coarse_data()
        output = model.run(50,50,3, content, styles, init)
        handler.write_image_output(output, content, styles, init)
    if False:
        content_images, style_images, init_images = handler.get_fine_data()
        output = model.run(50,50,3, content_images, style_images, init_images)
        import pdb; pdb.set_trace()
        output = handler.stich_images(output)
        write_image('test2.png', output)
    if False:
        content_images, style_images, init_images = handler.get_overlapping_data(overlap=10)
        update_fn = handler.get_update_function(overlap=10)

        output = model.run(50,50,3, content_images, style_images, init_images, update_fn)
        import pdb; pdb.set_trace()
        output = handler.stich_overlapping_images(output, overlap=10)
        write_image('test5.png', output)
    if False:
        device = '/device:CPU:0' if config['debug'] else "/device:GPU:0"
        with tf.Session() as sess, tf.device(device):
            model = CoarseFineModel(sess, config, 50, 50)
            tiles, style_tiles, init_tiles = handler.get_overlapping_data(zoom=2, overlap=4)
            update_fn = handler.get_update_function(overlap=4)
            output = model.stylize_patches(sess, tiles, style_tiles, init_tiles, update_fn)
            handler.write_overlapping_image_output(output, tiles, style_tiles, init_tiles, overlap=4)
