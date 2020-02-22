import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, Concatenate, Lambda


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.compat.v1.image.resize_bilinear, name='resize'):
    r""" resize feature map
    https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/layers.py#L114
    """
    if scale == 1:
        return x
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1] * scale, tf.int32),
                  tf.cast(xs[2] * scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)
    return x


def ATN_layer(input, ksize=3, stride=1, rate=2,
              softmax_scale=10., rescale=False):
    x1 = input[0]
    x2 = input[1]
    mask = input[2]
    # downsample input feature maps if needed due to limited GPU memory
    if rescale:
        x1 = resize(x1, scale=1. / 2, func=tf.image.resize_nearest_neighbor)
        x2 = resize(x2, scale=1. / 2, func=tf.image.resize_nearest_neighbor)
    # get shapes
    raw_x1s = tf.shape(x1)
    int_x1s = x1.get_shape().as_list()
    int_x2s = x2.get_shape().as_list()
    # extract patches from low-level feature maps for reconstruction
    kernel = 2 * rate
    raw_w = tf.extract_image_patches(
        x1, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')

    raw_w = tf.reshape(raw_w, [int_x1s[0], -1, kernel, kernel, int_x1s[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to [batch, kernel, kernel, channel, height/width]
    raw_w_groups = tf.split(raw_w, int_x1s[0], axis=0)
    # extract patches from high-level feature maps for matching and attending
    x2_groups = tf.split(x2, int_x2s[0], axis=0)
    w = tf.extract_image_patches(
        x2, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_x2s[0], -1, ksize, ksize, int_x2s[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to [b, ksize, ksize, c, hw/4]
    w_groups = tf.split(w, int_x2s[0], axis=0)
    # resize and extract patches from masks
    mask = resize(mask, to_shape=int_x2s[1:3], func=tf.image.resize_nearest_neighbor)

    # To avoid error with wrong mask shape(squeeze batch_size dimension to 1)
    mask_shape_1 = mask.get_shape().as_list()[0]
    if (mask_shape_1 > 1):
        mask = tf.split(mask, mask_shape_1, axis=0)[0]

    m = tf.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to [1, ksize, ksize, 1, hw/4]
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0, 1, 2], keep_dims=True), 0.), tf.float32)

    # matching and attending hole and non-hole patches
    y = []
    scale = softmax_scale
    for xi, wi, raw_wi in zip(x2_groups, w_groups, raw_w_groups):
        # matching on high-level feature maps
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")
        yi = tf.reshape(yi, [1, int_x2s[1], int_x2s[2], (int_x2s[1] // stride) * (int_x2s[2] // stride)])
        # apply softmax to obtain attention score
        yi *= mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask
        # transfer non-hole features into holes according to the atttention score
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_x1s[1:]], axis=0),
                                    strides=[1, rate * stride, rate * stride, 1]) / 4.
        y.append(yi)
    y = tf.concat(y, axis=0)
    y.set_shape(int_x1s)
    if rescale:
        y = resize(y, scale=2., func=tf.image.resize_nearest_neighbor)
    return y

def ATNConv(intx1_shape, intx2_shape, mask_shape, ksize=3, fuse=True):
    assert mask_shape[-1] == 1, "Mask can only have 1 channel!"
    input_x1 = Input(batch_shape=intx1_shape)
    input_x2 = Input(batch_shape=intx2_shape)
    input_mask = Input(batch_shape=mask_shape)
    x = Lambda(ATN_layer, arguments={"ksize": ksize})([input_x1, input_x2, input_mask])
    if fuse == True:
        fnum = intx1_shape[-1] // 4
        y1 = Conv2D(fnum, kernel_size=3, strides=1, dilation_rate=1, activation='relu', padding='SAME')(x)
        y2 = Conv2D(fnum, kernel_size=3, strides=1, dilation_rate=2, activation='relu', padding='SAME')(x)
        y3 = Conv2D(fnum, kernel_size=3, strides=1, dilation_rate=4, activation='relu', padding='SAME')(x)
        y4 = Conv2D(fnum, kernel_size=3, strides=1, dilation_rate=8, activation='relu', padding='SAME')(x)
        out = Concatenate(axis=3)([y1, y2, y3, y4])
        ATNConv_Model = Model([input_x1, input_x2, input_mask], out)
    else:
        ATNConv_Model = Model([input_x1, input_x2, input_mask], x)
    # ATNConv_Model.summary()
    # plot_model(ATNConv_Model, to_file='./Models/ATNConv.png')
    return ATNConv_Model
