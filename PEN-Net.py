import os
import matplotlib.pyplot as plt
import keras
import numpy as np
import math
from keras.utils import plot_model
from keras import Sequential, Input, Model
from keras.backend.numpy_backend import concatenate
from keras.engine.saving import load_model
from keras.layers import Conv2D, LeakyReLU, ReLU, UpSampling2D, Dense, Flatten, Dropout, BatchNormalization, \
    ZeroPadding2D
from keras import backend as K
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate
from keras.optimizers import Adam
from tensorflow.contrib.framework.python.ops import add_arg_scope
import scipy.misc
from keras import Input, Model
from keras.layers import Lambda, Conv2D, LeakyReLU, Activation
import keras.backend as K
import numpy as np
from batch_data_loader import *
import threading

import core.SpectralNormalization

class PENNet:
    def __init__(self):
        self.Is_trainning = True
        self.Is_using = False
        self.dataset_path = "C:/Users/qxdnf/PycharmProjects/tensorflow_xuexi1/PEN-Net-Keras/facade_imgs"
        self.mask_size = [128, 128]
        self.epochs = 1000000
        self.img_size = [256, 256, 3]
        self.batch_size = 1
        self.save_interval = 200
        self.mask_update_interval = 500
        self.optimizer = Adam(0.0001, 0.5, 0.999)
        print("Initializing Models.....")
        self.Config_GPU()
        previous_epoch = 0
        # used for ensure the output is valid for the discriminator
        self.Combined_Model, self.Generator, self.Discriminator = self.build_model(
            [(self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2] + 1),
             (self.batch_size, self.img_size[0], self.img_size[1], 1)])
        if os.path.exists("./Models/Discriminator_full.h5") and self.Is_using == True:
            print("Loading Saved Model weights.....")
            self.Discriminator = load_model("./Models/Discriminator_full.h5")
            self.Generator.load_weights("./Models/Generator_weights.h5")
            self.Combined_Model.load_weights("./Models/Combined_Model_weights.h5")
            print("Finished Loading Saved Model weights!")
        print("Compiling Models....")

        self.loss_weights = [1, 15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.Discriminator.compile(loss='mse',
                                   optimizer=Adam(0.0001, 0.5, 0.999),
                                   metrics=['accuracy'])
        self.Combined_Model.compile(loss=['mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mse'],
                                    loss_weights=self.loss_weights,
                                    optimizer=self.optimizer)

    def Avg_Output(map):
        batch_size = map.get_shape().as_list()[0]
        if batch_size == 1:
            output = tf.expand_dims(tf.expand_dims(tf.reduce_mean(map), axis=[0]), axis=-1)
        else:
            map_group = tf.split(map, batch_size, axis=0)
            output = tf.expand_dims(tf.reduce_mean(map_group[0]), axis=0)
            for i in range(batch_size - 1):
                output = tf.concat([output, tf.expand_dims(tf.reduce_mean(map_group[i + 1]), axis=0)], axis=0)
        return output

    def Build_Discriminator(input_shape, use_sigmoid=False, cnum=64):
        input = Input(batch_shape=input_shape)
        dkernel_size = 5
        x = SpectralNormalization(
            Conv2D(cnum, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(input)
        x = (LeakyReLU(alpha=0.2))(x)
        x = SpectralNormalization(
            Conv2D(cnum * 2, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(x)
        x = Conv2D(cnum * 2, kernel_size=3, strides=2, padding="same")(x)
        x = (LeakyReLU(alpha=0.2))(x)
        x = SpectralNormalization(
            Conv2D(cnum * 4, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        x = SpectralNormalization(
            Conv2D(cnum * 8, kernel_size=dkernel_size, strides=1, padding="same", use_bias=False))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        classifier = Conv2D(1, kernel_size=dkernel_size, strides=1, padding="same")

        if use_sigmoid == True:
            sig = Activation("relu")(x)
            out = classifier(sig)
        else:
            out = classifier(x)
        print("Discriminator Structure:")
        Discriminator = Model(input, out, name="Discriminator")
        Discriminator.summary()
        return Discriminator

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
        import tensorflow as tf
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

    def ATNConv(intx1_shape, intx2_shape, mask_shape, ksize=3, fuse=True):
        assert mask_shape[-1] == 1, "Mask can only have 1 channel!"
        input_x1 = Input(batch_shape=intx1_shape)
        input_x2 = Input(batch_shape=intx2_shape)
        input_mask = Input(batch_shape=mask_shape)
        x = Lambda(ATN_layer, arguments={"ksize": ksize})([input_x1, input_x2, input_mask])
        # out = DialConv_layer(intx1_shape)(x)
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

    def build_model(self, input_shape, cnum=32):
        img_input = Input(batch_shape=input_shape[0])
        mask_input = Input(batch_shape=input_shape[1])
        img_shape = [input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3] - 1]
        # encoder
        dw_conv01 = Sequential([Conv2D(cnum, kernel_size=3, strides=2, padding='same'),
                                LeakyReLU(0.2)])(img_input)
        dw_conv02 = Sequential([Conv2D(cnum * 2, kernel_size=3, strides=2, padding='same'),
                                (LeakyReLU(0.2))])(dw_conv01)
        dw_conv03 = Sequential([Conv2D(cnum * 4, kernel_size=3, strides=2, padding='same'),
                                (LeakyReLU(0.2))])(dw_conv02)
        dw_conv04 = Sequential([Conv2D(cnum * 8, kernel_size=3, strides=2, padding='same'),
                                (LeakyReLU(0.2))])(dw_conv03)
        dw_conv05 = Sequential([Conv2D(cnum * 16, kernel_size=3, strides=2, padding='same'),
                                (LeakyReLU(0.2))])(dw_conv04)
        dw_conv06 = Sequential([Conv2D(cnum * 16, kernel_size=3, strides=2, padding='same'),
                                (LeakyReLU(0.2))])(dw_conv05)
        dw_conv06_shape = dw_conv06.get_shape().as_list()
        dw_conv05_shape = dw_conv05.get_shape().as_list()
        dw_conv04_shape = dw_conv04.get_shape().as_list()
        dw_conv03_shape = dw_conv03.get_shape().as_list()
        dw_conv02_shape = dw_conv02.get_shape().as_list()
        dw_conv01_shape = dw_conv01.get_shape().as_list()
        mask_shape = mask_input.get_shape().as_list()

        at_conv05 = ATNConv(dw_conv05_shape, dw_conv06_shape, mask_shape, 1, False)(
            [dw_conv05, dw_conv06, mask_input])
        at_conv04 = ATNConv(dw_conv04_shape, dw_conv05_shape, mask_shape)([dw_conv04, dw_conv05, mask_input])
        at_conv03 = ATNConv(dw_conv03_shape, dw_conv04_shape, mask_shape)([dw_conv03, dw_conv04, mask_input])
        at_conv02 = ATNConv(dw_conv02_shape, dw_conv03_shape, mask_shape)([dw_conv02, dw_conv03, mask_input])
        at_conv01 = ATNConv(dw_conv01_shape, dw_conv02_shape, mask_shape)([dw_conv01, dw_conv02, mask_input])

        upx5_ = UpSampling2D((2, 2), interpolation='bilinear')(dw_conv06)
        up_conv05 = Conv2D(cnum * 16, kernel_size=3, strides=1, padding="same", activation='relu')(upx5_)
        upx4_ = UpSampling2D((2, 2), interpolation='bilinear')(Concatenate(axis=-1)([up_conv05, at_conv05]))
        up_conv04 = Conv2D(cnum * 8, kernel_size=3, strides=1, padding="same", activation='relu')(upx4_)
        upx3_ = UpSampling2D((2, 2), interpolation='bilinear')(Concatenate(axis=-1)([up_conv04, at_conv04]))
        up_conv03 = Conv2D(cnum * 4, kernel_size=3, strides=1, padding="same", activation='relu')(upx3_)
        upx2_ = UpSampling2D((2, 2), interpolation='bilinear')(Concatenate(axis=-1)([up_conv03, at_conv03]))
        up_conv02 = Conv2D(cnum * 2, kernel_size=3, strides=1, padding="same", activation='relu')(upx2_)
        upx1_ = UpSampling2D((2, 2), interpolation='bilinear')(Concatenate(axis=-1)([up_conv02, at_conv02]))
        up_conv01 = Conv2D(cnum, kernel_size=3, strides=1, padding="same", activation='relu')(upx1_)

        torgb5 = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='tanh')
        torgb4 = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='tanh')
        torgb3 = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='tanh')
        torgb2 = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='tanh')
        torgb1 = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='tanh')

        img5 = torgb5(Concatenate(axis=-1)([up_conv05, at_conv05]))
        img4 = torgb4(Concatenate(axis=-1)([up_conv04, at_conv04]))
        img3 = torgb3(Concatenate(axis=-1)([up_conv03, at_conv03]))
        img2 = torgb2(Concatenate(axis=-1)([up_conv02, at_conv02]))
        img1 = torgb1(Concatenate(axis=-1)([up_conv01, at_conv01]))

        output_ = UpSampling2D((2, 2), interpolation='bilinear')(Concatenate(axis=-1)([up_conv01, at_conv01]))

        decoded_out = Sequential([
            Conv2D(cnum * 2, kernel_size=3, strides=1, padding="same", activation='relu'),
            Conv2D(3, kernel_size=3, strides=1, padding="same", activation='tanh')])(output_)
        mask_out = Lambda(Mask_Img_layer)([decoded_out, mask_input])
        unmask_out = Lambda(UnMasked_Img_layer)([decoded_out, mask_input])
        Discriminator = Build_Discriminator(img_shape)
        Discriminator.compile(loss='binary_crossentropy',
                              optimizer=Adam(0.0002, 0.5, 0.999),
                              metrics=['accuracy'])
        Discriminator.trainable = False  # Combined Model's Discriminator can not be trained
        discrim_out = Discriminator(decoded_out)

        Generator = Model([img_input, mask_input],
                          [decoded_out, mask_out, unmask_out, img1, img2, img3, img4, img5], name="Generator")
        Combined_Model = Model([img_input, mask_input],
                               [mask_out, unmask_out, img1, img2, img3, img4, img5, discrim_out],
                               name="Combined_Model")

        print("Generator Structure:")
        Generator.summary()
        print("Combined_Model Structure:")
        Combined_Model.summary()
        return Combined_Model, Generator, Discriminator

    def Generated_Img(Img, Masks):
        # if Masks.shape[0]>1:
        #     Single_Mask=Masks[0:1,:,:,:]
        # else:
        #     Single_Mask=Masks
        Masked_Imgs = Masked_Img(Img, Masks)
        Org_Predicted_Imgs = Generator.predict([Img2Img_with_mask(Masked_Imgs, Masks), Masks])[0]
        UnMasked_Imgs = Img * (1 - Masks)
        Output = Org_Predicted_Imgs * Masks + UnMasked_Imgs
        # plt.imshow(Org_Predicted_Imgs[0])
        # plt.figure()
        # plt.imshow(UnMasked_Imgs[0])
        # plt.figure()
        # plt.imshow((Org_Predicted_Imgs*Masks)[0])
        # plt.show()
        return Output
        # return Org_Predicted_Imgs

    def Masked_Img(Img, Masks):
        # keep axis[-1]=3
        return np.clip((Img + Masks), 0, 1)

    def Img2Img_with_mask(Img, Masks):
        # convert axis[-1]=3 to axis[-1]=4
        return np.concatenate([Img, Masks], axis=-1)

    def Load_data_norm(batch_size):
        # Batch_Img = Load_data(batch_size,dataset_path,[256,256])
        Batch_Img = Load_from_rawdata(batch_size)
        return img255_normalization(Batch_Img)

    def Random_Rectangle_Mask(img, shape=[128, 128], Is_Centred=False, Is_Random_size=True, Is_square=True,
                              Maximum_smaller_scale=1.5):
        # Generate ramdom position
        batch, x, y, _ = img.shape
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
        batch_mask = np.zeros([batch, x, y, 1])
        if Is_Random_size == True:
            mask_shape = []
            mask_shape.append(np.random.randint(int(shape[0] / Maximum_smaller_scale), shape[0]))
            if Is_square == True:
                mask_shape.append(mask_shape[0])
            else:
                mask_shape.append(np.random.randint(int(shape[1] / Maximum_smaller_scale), shape[1]))
        else:
            mask_shape = shape
        batch_mask[:, position[0] - mask_shape[0] // 2:position[0] + mask_shape[0] // 2,
        position[1] - mask_shape[1] // 2:position[1] + mask_shape[1] // 2, :] = 1
        img_mask = Masked_Img(img, batch_mask)
        return img_mask, batch_mask

    def Centre_Mask(img, shape=[128, 128]):
        batch, x, y, _ = img.shape
        batch_mask = np.zeros([batch, x, y, 1])
        mask = np.zeros([x, y, 1])
        single_mask = np.zeros([1, x, y, 1])
        xc = x // 2
        yc = y // 2
        mask[xc - shape[0] // 2:xc + shape[0] // 2, yc - shape[1] // 2:yc + shape[1] // 2, :] = 1
        batch_mask[:, xc - shape[0] // 2:xc + shape[0] // 2, yc - shape[1] // 2:yc + shape[1] // 2, :] = 1
        single_mask[:, xc - shape[0] // 2:xc + shape[0] // 2, yc - shape[1] // 2:yc + shape[1] // 2, :] = 1
        img_mask = Masked_Img(img, batch_mask)
        return img_mask, batch_mask

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def img255_normalization(img):
        n_img = (img / 255)
        return n_img

    def save_imgs(epcho, mask_size):
        name_g = "generated_epoch_" + str(epcho) + ".png"
        name_decoding = "decoding_epoch_" + str(epcho) + ".png"
        name_real_out = "real_output_epoch_" + str(epcho) + ".png"
        path_g = "./Generated_Imgs/" + name_g
        path_decoding = "./Generated_Imgs/Decoding_Imgs/" + name_decoding
        path_realout = "./Generated_Imgs/Real_Output_Imgs/" + name_real_out
        if not os.path.exists("./Generated_Imgs"):  # 如果路径不存在
            os.makedirs("./Generated_Imgs")
        if not os.path.exists("./Generated_Imgs/Decoding_Imgs"):  # 如果路径不存在
            os.makedirs("./Generated_Imgs/Decoding_Imgs")
        if not os.path.exists("./Generated_Imgs/Real_Output_Imgs/"):  # 如果路径不存在
            os.makedirs("./Generated_Imgs/Real_Output_Imgs/")
        Batch_Img = Load_data_norm(batch_size)
        # Masked_Img_Batch, Masks = Centre_Mask(Batch_Img, mask_size)
        Masked_Img_Batch, Masks = Random_Rectangle_Mask(Batch_Img, mask_size)
        # Real_Output=Generated_Img(Batch_Img,Masks)
        Output_Img, Masked_Img, UnMasked_Img, Img1, Img2, Img3, Img4, Img5 = Generator.predict(
            [Img2Img_with_mask(Masked_Img_Batch, Masks), Masks])
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(normalization(Masked_Img_Batch[0]))
        plt.title('Masked')
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(normalization(Batch_Img[0]))
        plt.title('GT')
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(normalization(Generated_Img(Batch_Img, Masks)[0]))
        plt.title('PEN-Net Improved')
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.imshow(Output_Img[0])
        plt.title('PEN-Net Out')
        plt.axis("off")
        plt.savefig(path_g)
        plt.close()
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(normalization(Output_Img[0]))
        plt.title('G_Out')
        plt.axis("off")
        plt.subplot(2, 4, 2)
        plt.imshow(normalization(Img1[0]))
        plt.title('ATN5+UP5')
        plt.axis("off")
        plt.subplot(2, 4, 3)
        plt.imshow(normalization(Img2[0]))
        plt.title('ATN4+UP4')
        plt.axis("off")
        plt.subplot(2, 4, 4)
        plt.imshow(normalization(Img3[0]))
        plt.title('ATN3+UP3')
        plt.axis("off")
        plt.subplot(2, 4, 5)
        plt.imshow(normalization(Img4[0]))
        plt.title('ATN2+UP2')
        plt.axis("off")
        plt.subplot(2, 4, 6)
        plt.imshow(normalization(Img5[0]))
        plt.title('ATN1+UP1')
        plt.axis("off")
        plt.subplot(2, 4, 7)
        plt.imshow(Masked_Img[0])
        plt.title('Masked')
        plt.axis("off")
        plt.subplot(2, 4, 8)
        plt.imshow(UnMasked_Img[0])
        plt.title('UnMasked')
        plt.axis("off")
        plt.savefig(path_decoding)
        plt.close()
        plt.imsave(path_realout, np.clip(Output_Img[0], 0, 1), format="png")

    def save_model():
        if not os.path.exists("./Models"):  # 如果路径不存在
            os.makedirs("./Models")
        Generator.save_weights('./Models/Generator_weights.h5')
        Discriminator.save('./Models/Discriminator_full.h5')
        Combined_Model.save_weights('./Models/Combined_Model_weights.h5')

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

    def train(epochs, batch_size, mask_size, save_interval=100, mask_update_interval=1000, acc_mask_size=2):
        print("Start_training....")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        Initialize_Model()
        valid = np.ones(shape=[batch_size, 16, 16, 1])
        fake = np.zeros(shape=[batch_size, 16, 16, 1])
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))
        epoch_list = []
        G_loss_list = []
        # follow multi-thread approach to read data
        Batch_Img = Load_data_norm(batch_size)
        # Real_img = Load_data_norm(batch_size)
        for epoch in range(epochs):
            epoch = epoch + previous_epoch
            try:
                if Batch_Img == None:
                    Batch_Img = Load_data_norm(batch_size)
            except:
                pass
            load_thread1 = LoadingThread(eval('Load_data_norm'), args=(batch_size,))
            load_thread1.start()
            # load_thread2 = LoadingThread(eval('Load_data_norm'), args=(batch_size,))
            # load_thread2.start()
            # train Discriminator

            # useless when using the multi-thread
            # Batch_Img = Load_data_norm(batch_size)
            # Real_img=Load_data_norm(batch_size)

            # use a advanced model to generate mask instead of the simple centre mask one
            Masked_Img_Batch, Masks = Random_Rectangle_Mask(Batch_Img, mask_size)
            # Masked_Img_Batch, Masks=Centre_Mask(Batch_Img, mask_size)
            Fake_img = Generated_Img(Masked_Img_Batch, Masks)
            # Fake_img=Generator.predict(Img2Img_with_mask(Masked_Img_Batch,Masks), Masks)[0]
            Resized_Imgs = Pyramidal_Img_Resize(Batch_Img)

            # D_loss_real = Discriminator.train_on_batch(Batch_Img, valid)
            D_loss_real = Discriminator.train_on_batch(Batch_Img, valid)
            D_loss_fake = Discriminator.train_on_batch(Fake_img, fake)
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
            # train Generator
            G_loss = Combined_Model.train_on_batch([Img2Img_with_mask(Masked_Img_Batch, Masks), Masks],
                                                   [(Resized_Imgs[0] * (1 - Masks)), (Resized_Imgs[0] * Masks),
                                                    Resized_Imgs[1], Resized_Imgs[2],
                                                    Resized_Imgs[3], Resized_Imgs[4], Resized_Imgs[5],
                                                    valid])
            G_loss_list.append(G_loss[6])

            print("%d [D loss: %f, acc.: %.2f%%] [G L1 loss: %f , Adv loss: %f]" % (
                epoch, D_loss[0], 100 * D_loss[1], G_loss[0], G_loss[6]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print(np.mean(Discriminator.predict(Load_data_norm(1))))
                print("Saving Models....")
                save_imgs(epoch, mask_size)
                if epoch != 0:
                    save_model()
            if epoch % mask_update_interval == 0 and epoch != 0 and acc_mask_size >= 0:
                mask_size[0] = mask_size[0] + acc_mask_size
                mask_size[1] = mask_size[1] + acc_mask_size
                if mask_size[0] >= 128:
                    mask_size = [128, 128]

            Batch_Img = load_thread1.get_result()
            # Real_img = load_thread2.get_result()

    def Initialize_Model():
        # To prevent the "FailedPreconditionError"
        Batch_Img_Initialize = img255_normalization(np.ones([batch_size, img_size[0], img_size[1], img_size[2]]))
        Masked_Img_Batch_Initialize, Masks_Initialize = Centre_Mask(Batch_Img_Initialize, mask_size)
        Generated_Img(Masked_Img_Batch_Initialize, Masks_Initialize)[0]

    def Config_GPU():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
