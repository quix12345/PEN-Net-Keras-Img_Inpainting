import tensorflow as tf
import numpy as np
import scipy.misc

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def img255_normalization(img):
    n_img = (img / 255)
    return n_img

def UnMasked_Img_layer(input):
    Img = input[0]
    Mask = input[1]
    Mask = tf.concat([Mask, Mask, Mask], axis=-1)
    return tf.multiply(Img, Mask)

def Mask_Img_layer(input):
    Img = input[0]
    Mask = input[1]
    sMask = tf.subtract(tf.constant(1., shape=Mask.get_shape().as_list()), Mask)
    sMask = tf.concat([sMask, sMask, sMask], axis=-1)
    return tf.multiply(Img, sMask)

def Masked_Img(Img, Masks):
    # keep axis[-1]=3
    return np.clip((Img + Masks), 0, 1)

def Img2Img_with_mask(Img, Masks):
    # convert axis[-1]=3 to axis[-1]=4
    return np.concatenate([Img, Masks], axis=-1)

def Pyramidal_Img_Resize(Batch_Img, pyramidal_img_size=[256, 128, 64, 32, 16, 8]):
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