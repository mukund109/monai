def read_image(path):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img

def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)

def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)

def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[...,::-1]
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
    # rgb to bgr
    imgpost = imgpost[...,::-1]
    return imgpost

def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            flow[0,y,x] = struct.unpack('f', f.read(4))[0]
            flow[1,y,x] = struct.unpack('f', f.read(4))[0]
    return flow


def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else: return [0.] * len(weights)

def get_content_image(content_img):
    path = os.path.join(config['content_img_dir'], content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    h, w, d = img.shape

    #resize content image to size of required output image
    h, w = int(h*config['output_scale']), int(w*config['output_scale'])
    img = cv2.resize(img, dsize=(h, w), interpolation = cv2.INTER_LANCZOS4)

    # mx = config['max_size']
    # # resize if > max size
    # if h > w and h > mx:
    #     w = (float(mx) / float(h)) * w
    #     img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    # if w > mx:
    #     h = (float(mx) / float(w)) * h
    #     img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img

def get_style_images(content_img):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in config['style_imgs']:
        path = os.path.join(config['style_imgs_dir'], style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs

def get_init_image(init_type, content_img, style_imgs, frame=None, init_img_path=None):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(args.noise_ratio, content_img)
        return init_img
    elif init_type == 'custom':
        init_img = cv2.imread(init_img_path, cv2.IMREAD_COLOR)
        check_image(init_img, init_img_path)
        init_img = init_img.astype(np.float32)
        _, h, w, _ = content_img.shape
        init_img = cv2.resize(init_img, dsize=(int(h), int(w)), interpolation=cv2.INTER_AREA)
        return preprocess(init_img)
    else:
        raise FileNotFoundError()
