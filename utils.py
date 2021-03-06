import cv2
import numpy as np
import json
from collections import OrderedDict
import os

def get_config():
    json_str = ''
    with open('config.json', 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    return opt

def check_image(img, path):
    if img is None:
        raise FileNotFoundError("No such file {}".format(path))

def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_noise_image(noise_ratio, content_img, seed):
    np.random.seed(seed)
    noise_img = np.random.uniform(90., 160., content_img.shape)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img.astype('uint8')

def image_to_tiles(image, zoom):
    assert image.shape[0]==image.shape[1]
    assert image.shape[0]%zoom == 0
    tile_size = image.shape[0]//zoom
    tiles = []
    for i in range(zoom):
        for j in range(zoom):
            tile = image[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size]
            tiles.append(tile)
    return tiles

def tiles_to_image(tiles):
    num = len(tiles)
    assert (num**(1/2)-int(num**(1/2)))==0
    num_ = int(num**(1/2))

    tile_size = tiles[0].shape[0]
    image = np.ndarray(shape=(num_*tile_size, num_*tile_size, 3), dtype='uint8')

    for i in range(num_):
        for j in range(num_):
            image[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size] = tiles[i*num_+j]

    return image

def image_to_overlapping_tiles(image, zoom, overlap):
    base_size = (image.shape[0] + overlap*(zoom-1))//zoom - overlap
    tiles = []
    for i in range(zoom):
        for j in range(zoom):
            tile = image[i*base_size:(i+1)*base_size+overlap,j*base_size:(j+1)*base_size+overlap]
            tiles.append(tile)
    return tiles

def overlapping_tiles_to_image(tiles, overlap, bring_to_front_index=None):
    zoom = len(tiles)
    assert (zoom**(1/2)-int(zoom**(1/2)))==0
    zoom = int(len(tiles)**(1/2))

    new_size = (tiles[0].shape[0]*zoom - overlap*(zoom-1))
    base_size = (new_size + overlap*(zoom-1))//zoom - overlap

    image =  np.ndarray(shape=(new_size, new_size, 3), dtype='uint8')
    for i in range(zoom):
        for j in range(zoom):
            image[i*base_size:(i+1)*base_size+overlap,j*base_size:(j+1)*base_size+overlap] = tiles[i*zoom + j]

    if bring_to_front_index is not None:
        k = bring_to_front_index
        i,j = k//zoom, k%zoom
        image[i*base_size:(i+1)*base_size+overlap,j*base_size:(j+1)*base_size+overlap] = tiles[k]

    return image


def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else: return [0.] * len(weights)


def preprocess(img):
    imgpre = np.copy(img)
    imgpre = imgpre.astype(np.float32)
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis,:,:,:]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return imgpre

def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    return imgpost

def read_image(path):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    # bgr to rgb
    img = img[...,::-1]
    return img

def write_image(path, img):
    # rgb to bgr
    img = img[...,::-1]
    cv2.imwrite(path, img)

class ImageHandler(object):
    def __init__(self, config, base_size=None):
        self.base_size = base_size
        # i hate tensorflow
        assert self.base_size[0] ==self.base_size[1]
        self.config = config

        #Images are stores in original resolution
        #Read content images
        self.content_image = read_image(self.config['content_image'])

        #Read style images
        self.style_images = []
        for path in self.config['style_images']:
            self.style_images.append(read_image(path))
        #Read init_image
        self.init_image = self.get_init_image(self.config['init_image_type'], self.content_image, self.style_images, self.config['init_image_path'])

    def get_coarse_data(self):
        content = preprocess(cv2.resize(self.content_image, dsize=self.base_size, interpolation=cv2.INTER_LANCZOS4))
        styles = []
        for img in self.style_images:
            styles.append(preprocess(cv2.resize(img, dsize=self.base_size, interpolation=cv2.INTER_LANCZOS4)))
        init = preprocess(cv2.resize(self.init_image, dsize=self.base_size, interpolation=cv2.INTER_LANCZOS4))
        return content, styles, init

    def get_fine_data(self, zoom=2, init_image=None):

        new_size = (self.base_size[0]*zoom, self.base_size[1]*zoom)

        content = cv2.resize(self.content_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        content_crops = image_to_tiles(content, zoom)
        content_crops = list(map(preprocess, content_crops))

        style_crops = []
        for simg in self.style_images:
            simg = cv2.resize(simg, new_size, interpolation=cv2.INTER_LANCZOS4)
            simg_crops = image_to_tiles(simg, zoom)
            simg_crops = list(map(preprocess, simg_crops))
            style_crops.append(simg_crops)

        if init_image is None:
            init_image = self.init_image
        init_image = cv2.resize(init_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        init_crops = image_to_tiles(init_image, zoom)
        init_crops = list(map(preprocess, init_crops))

        return content_crops, style_crops, init_crops

    def stich_images(self, images):
        images = list(map(postprocess, images))
        return tiles_to_image(images)

    def stich_overlapping_images(self, images, overlap, bring_to_front_index=None):
        images = list(map(postprocess, images))
        return overlapping_tiles_to_image(images, overlap, bring_to_front_index)

    def get_overlapping_data(self, zoom=2, overlap=100 , init_image=None):
        new_size = (self.base_size[0]*zoom - overlap*(zoom-1))
        new_size = (new_size, new_size)

        content = cv2.resize(self.content_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        content_crops = image_to_overlapping_tiles(content, zoom, overlap)
        content_crops = list(map(preprocess, content_crops))

        style_crops = []
        for simg in self.style_images:
            simg = cv2.resize(simg, new_size, interpolation=cv2.INTER_LANCZOS4)
            simg_crops = image_to_overlapping_tiles(simg, zoom, overlap)
            simg_crops = list(map(preprocess, simg_crops))
            style_crops.append(simg_crops)

        if init_image is None:
            init_image = self.init_image
        init_image = cv2.resize(init_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        init_crops = image_to_overlapping_tiles(init_image, zoom, overlap)
        init_crops = list(map(preprocess, init_crops))

        return content_crops, style_crops, init_crops

    def get_update_function(self, overlap):
        def update_overlapping_image(tile, index, all_tiles):
            all_tiles[index] = tile
            image = self.stich_overlapping_images(all_tiles, overlap, index)

            zoom = int(len(all_tiles)**(1/2))
            all_tiles = image_to_overlapping_tiles(image, zoom, overlap)
            all_tiles = list(map(preprocess, all_tiles))
            return all_tiles
        return update_overlapping_image

    def save_checkpoint(checkpoint_img, iteration):
        maybe_make_directory('checkpoints')
        write_image('checkpoints/'+'checkpoint'+str(iteration)+'.png', checkpoint_img)

    def convert_to_original_colors(content_img, stylized_img):
        content_img  = postprocess(content_img)
        stylized_img = postprocess(stylized_img)
        if config['color_convert_type'] == 'yuv':
            cvt_type = cv2.COLOR_BGR2YUV
            inv_cvt_type = cv2.COLOR_YUV2BGR
        elif config['color_convert_type'] == 'ycrcb':
            cvt_type = cv2.COLOR_BGR2YCR_CB
            inv_cvt_type = cv2.COLOR_YCR_CB2BGR
        elif config['color_convert_type'] == 'luv':
            cvt_type = cv2.COLOR_BGR2LUV
            inv_cvt_type = cv2.COLOR_LUV2BGR
        elif config['color_convert_type'] == 'lab':
            cvt_type = cv2.COLOR_BGR2LAB
            inv_cvt_type = cv2.COLOR_LAB2BGR
        content_cvt = cv2.cvtColor(content_img, cvt_type)
        stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
        c1, _, _ = cv2.split(stylized_cvt)
        _, c2, c3 = cv2.split(content_cvt)
        merged = cv2.merge((c1, c2, c3))
        dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
        dst = preprocess(dst)
        return dst

    def get_init_image(self, init_type, content_img, style_imgs, init_img_path=None):
        if init_type == 'content':
            return content_img
        elif init_type == 'style':
            return style_imgs[0]
        elif init_type == 'random':
            init_img = get_noise_image(self.config['noise_ratio'], content_img, self.config['random_seed'])
            return init_img
        elif init_type == 'custom':
            init_img = read_image(init_img_path)
            return init_img
        else:
            raise FileNotFoundError()

    def write_overlapping_image_output(self, output, content_tiles, style_tiles, init_tiles, overlap):

        output, content, init = [self.stich_overlapping_images(tiles, overlap) for tiles in  [output, content_tiles, init_tiles]]
        style_imgs = [self.stich_overlapping_images(tiles, overlap) for tiles in style_tiles]
        out_dir = self.config["image_output_dir"]
        maybe_make_directory(out_dir)
        img_path = os.path.join(out_dir,  "output.png")
        content_path = os.path.join(out_dir, 'content.png')
        init_path = os.path.join(out_dir, 'init.png')

        write_image(img_path, output)
        write_image(content_path, content)
        write_image(init_path, init)
        index = 0
        for style_img in style_imgs:
            path = os.path.join(out_dir, 'style_'+str(index)+'.png')
            write_image(path, style_img)
            index += 1

    def write_image_output(self, output_img, content_img, style_imgs, init_img):
        out_dir = self.config["image_output_dir"]
        maybe_make_directory(out_dir)
        img_path = os.path.join(out_dir,  "output.png")
        content_path = os.path.join(out_dir, 'content.png')
        init_path = os.path.join(out_dir, 'init.png')

        write_image(img_path, postprocess(output_img))
        write_image(content_path, postprocess(content_img))
        write_image(init_path, postprocess(init_img))
        index = 0
        for style_img in style_imgs:
            path = os.path.join(out_dir, 'style_'+str(index)+'.png')
            write_image(path, postprocess(style_img))
            index += 1

        # save the configuration settings
        out_file = os.path.join(out_dir, 'meta_data.json')
        with open(out_file, 'w') as f:
            json.dump(self.config, f)

if __name__=='__main__':
    from utils import get_config
    import matplotlib.pyplot as plt
    config = get_config()
    handler = ImageHandler(config, base_size=(500,500))
    a = handler.get_overlapping_data(zoom=3, overlap=200)
    import pdb; pdb.set_trace()
    for i,img in enumerate(a[0]):
        write_image('output{}.png'.format(i), postprocess(img))

    update = handler.get_update_function(overlap=200)

    tile = a[0][2]
    tile = tile*0
    b = update(tile, 2, a[0])
    write_image('1.png', postprocess(b[1]))
