import tensorflow as tf
import numpy as np
import scipy.misc


def normalization(data):
    """
    normalized the input data
    :param data: input
    :return: normalized data
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def img255_normalization(img):
    """
    Standard img normalization function in the project
    :param img: input img
    :return: normalized img
    """
    n_img = (img / 255)
    return n_img


def UnMasked_Img_layer(input):
    """
    Keras Lambda layer for mask process
    :param input: Raw output tensor from the Generator
    :return: Unmasked output tensor
    """
    Img = input[0]
    Mask = input[1]
    Mask = tf.concat([Mask, Mask, Mask], axis=-1)
    return tf.multiply(Img, Mask)


def Mask_Img_layer(input):
    """
       Keras Lambda layer for mask process
       :param input: Raw output tensor from the Generator
       :return: Masked output tensor
       """
    Img = input[0]
    Mask = input[1]
    sMask = tf.subtract(tf.constant(1., shape=Mask.get_shape().as_list()), Mask)
    sMask = tf.concat([sMask, sMask, sMask], axis=-1)
    return tf.multiply(Img, sMask)


def Masked_Img(Img, Masks):
    """
    Generate the masked img
    :param Img: input imgs
    :param Masks: masks
    :return: masked imgs
    """
    # keep axis[-1]=3
    return np.clip((Img + Masks), 0, 1)


def Img2Img_with_mask(Img, Masks):
    """
    Covert img to img_with_mask
    :param Img: input imgs
    :param Masks: input masks
    :return: imgs with masks
    """
    # convert axis[-1]=3 to axis[-1]=4
    return np.concatenate([Img, Masks], axis=-1)


def Pyramidal_Img_Resize(Batch_Img, pyramidal_img_size=[256, 128, 64, 32, 16, 8]):
    """
    Resize the input Img to pyramidal imgs
    :param Batch_Img: input imgs
    :param pyramidal_img_size: list of different size for pyramidal imgs
    :return: Pyramidal imgs
    """
    batch_num = Batch_Img.shape[0]
    i = 0
    # decoder,batch,h,w,c
    Batch_Pyramidal_Resized_Img = []
    while i < len(pyramidal_img_size):
        j = 0
        Pyramidal_Resized_Img = []
        while j < batch_num:
            Pyramidal_Resized_Img.append(img255_normalization(
                scipy.misc.imresize(Batch_Img[j], [pyramidal_img_size[i], pyramidal_img_size[i]])))
            j = j + 1
        i = i + 1
        Batch_Pyramidal_Resized_Img.append(np.array(Pyramidal_Resized_Img))
    return Batch_Pyramidal_Resized_Img


def img_brighten(img, factor):
    """
    Increase the brightness of the input img
    :param img: input img
    :param factor: the degree of increasing the brightness
    :return: output img
    """
    img2 = img * factor
    img2 = np.clip(img2, 0, 255)
    return img2


def img_noise(img, factor=0.02):
    """
    Add noise to the img
    :param img: input img
    :param factor: noise factor
    :return: output img
    """
    img_noisy = img + factor * np.random.normal(0, 255, img.shape)
    img_noisy = np.clip(img_noisy, 0, 255)
    return img_noisy


def img2rgb(img):
    """
    Convert the image into rgb mode for displaying
    :param img: input img
    :return: output img
    """
    img_rgb = (img / 255)[:, :, [2, 1, 0]]
    return img_rgb


def img_reversed(img, mode):
    """
    Reverse the img on different directions
    :param img: input img
    :param mode: reverse mode
    :return: new img
    """
    x_size, y_size, _ = img.shape
    img_reverse = np.zeros([x_size, y_size, 3])
    if mode == 1:
        i = 0
        while i < x_size:
            img_reverse[i] = img[-i]
            i = i + 1
    elif mode == 2:
        i = 0
        while i < y_size:
            img_reverse[:, i] = img[:, -i]
            i = i + 1
    elif mode == 0:
        img_reverse = img
    return img_reverse


def Random_Rectangle_Img(img, shape=[256, 256], Is_Centred=False, Is_Random_size=False, Is_square=True,
                         Maximum_smaller_scale=3):
    """
    Generate Mask at ramdom position
    :param img: input img
    :param shape: Mask shape
    :param Is_Centred: whether the mask is centred
    :param Is_Random_size: whether the mask used a random size
    :param Is_square: whether the mask is a square
    :param Maximum_smaller_scale: A config to generate random size masks,default to be 3
    :return: Img in the mask
    """
    x, y, _ = img.shape
    position = []
    if Is_Centred == True:
        xc = x // 2
        yc = y // 2
        position.append(xc)
        position.append(yc)
    else:
        xu_limit = x - shape[0] // 2
        xl_limit = shape[0] // 2
        yu_limit = y - shape[1] // 2
        yl_limit = shape[1] // 2
        position.append(np.random.randint(xl_limit, xu_limit))
        position.append(np.random.randint(yl_limit, yu_limit))
    # Generate masks
    if Is_Random_size == True:
        mask_shape = []
        mask_shape.append(np.random.randint(shape[0] // Maximum_smaller_scale, shape[0]))
        if Is_square == True:
            mask_shape.append(mask_shape[0])
        else:
            mask_shape.append(np.random.randint(shape[1] // Maximum_smaller_scale, shape[1]))
    else:
        mask_shape = shape
    extract_img = img[position[0] - mask_shape[0] // 2:position[0] + mask_shape[0] // 2,
                  position[1] - mask_shape[1] // 2:position[1] + mask_shape[1] // 2, :]
    return np.array(extract_img, dtype=int)


def Resize_Img(img, ll_scale_factor=1.5):
    """
    Resize the Img
    :param img: input Img
    :param ll_scale_factor:lowest scale factor
    :return: resized Img
    """
    x, y, _ = img.shape
    if x >= y:
        ul_scale = y / 260
    else:
        ul_scale = x / 260
    ll_scale = ul_scale / ll_scale_factor
    scale = np.random.uniform(ll_scale, ul_scale)
    new_img_size = [int(x / scale), int(y / scale), _]

    return scipy.misc.imresize(img, new_img_size)


def GenerateValidInputImg(img,lower_limit = 220):
    """
    Used to generate valid input for the PEN-Net for testing
    :param img: raw data
    :param lower_limit: mask judgement factor
    :return: valid input for PEN-Net
    """
    mask_postions = []
    img_size = [256, 256]
    batch_size = 8
    r = 0
    while r < img_size[0]:
        l = 0
        while l < img_size[1]:
            if (img[r][l][0] >= lower_limit and img[r][l][1] >= lower_limit and img[r][l][2] >= lower_limit):
                mask_postions.append([r, l, 0])
            l = l + 1
        r = r + 1

    mask = np.zeros(shape=[img_size[0], img_size[1], 1])
    for mask_position in mask_postions:
        mask[mask_position[0], mask_position[1], 0] = 1
    input_img = np.zeros(shape=[batch_size, img_size[0], img_size[1], 3])
    input_masks = np.zeros(shape=[batch_size, img_size[0], img_size[1], 1])
    for i in range(batch_size):
        input_img[i, :, :, :] = img
        input_masks[i, :, :, :] = mask
    input_img = np.concatenate([input_img, input_masks], axis=-1)
    return input_img / 255, input_masks
