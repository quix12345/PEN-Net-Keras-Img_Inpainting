import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model
from keras import Sequential, Input, Model
from keras.engine.saving import load_model
from keras.layers import Concatenate
from keras.optimizers import Adam
import scipy.misc
from keras.layers import Lambda, Conv2D, LeakyReLU, Activation, UpSampling2D
from core.multi_thread import *
from core import SpectralNormalization as sp
from core.ATN_layer import ATNConv
from core.mask_processing import *
from core.utils import *
from core.data_loader import *


class PENNet:
    def __init__(self):
        """
        Initialize the whole PEN-Net!
        """
        self.Is_Auto_Loading = True
        self.Is_Plot_Model = True
        self.Is_debug_discri = False
        self.dataset_path = "./dataset"
        self.epochs = 1000000
        self.mask_size = [30, 30]
        self.img_size = [256, 256, 3]
        self.batch_size = 1
        self.save_interval = 200
        self.acc_mask_size = 5
        self.mask_update_interval = 500
        self.optimizer = Adam(0.0001, 0.5, 0.999)
        self.loss_weights = [1, 6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.previous_epoch = 0
        self.data_loader = data_loader(self.dataset_path, self.batch_size)

        # used for ensure the output is valid for the discriminator
        self.Combined_Model, self.Generator, self.Discriminator = self.build_model(
            [(self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2] + 1),
             (self.batch_size, self.img_size[0], self.img_size[1], 1)])
        if os.path.exists("./models/Discriminator_full.h5") and self.Is_Auto_Loading:
            print("Loading Saved Model weights.....")
            self.Discriminator = load_model("./models/Discriminator_full.h5")
            self.Generator.load_weights("./models/Generator_weights.h5")
            self.Combined_Model.load_weights("./models/Combined_Model_weights.h5")
            print("Finished Loading Saved Model weights!")
            if os.path.exists("./models/epoch.txt"):
                with open("./models/epoch.txt", "r") as f:
                    self.previous_epoch = int(f.readline())

        print("Compiling models....")

        if self.Is_Plot_Model:
            pass
            if not os.path.exists("./models/Model_Structure"):  # 如果路径不存在
                os.makedirs("./models/Model_Structure")
            plot_model(self.Combined_Model, to_file='./models/Model_Structure/Combined_Model.png')
            plot_model(self.Generator, to_file='./models/Model_Structure/Generator.png')
            plot_model(self.Discriminator, to_file='./models/Model_Structure/Discriminator.png')

        self.Discriminator.compile(loss='mae',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.Combined_Model.compile(loss=['mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae'],
                                    loss_weights=self.loss_weights,
                                    optimizer=self.optimizer)
        print("Initializing Models.....")
        self.Initialize_Model()

    def build_discriminator(self, input_shape, use_sigmoid=False, cnum=64):
        """
        Build up the Markovian Discriminator
        :param input_shape: Input shape
        :param use_sigmoid: whether use the sigmoid activation function at the end
        :param cnum: channel number
        :return:
        """
        input = Input(batch_shape=input_shape)
        dkernel_size = 5
        x = sp.SpectralNormalization(
            Conv2D(cnum, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(input)
        x = (LeakyReLU(alpha=0.2))(x)
        x = sp.SpectralNormalization(
            Conv2D(cnum * 2, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(x)
        x = Conv2D(cnum * 2, kernel_size=3, strides=2, padding="same")(x)
        x = (LeakyReLU(alpha=0.2))(x)
        x = sp.SpectralNormalization(
            Conv2D(cnum * 4, kernel_size=dkernel_size, strides=2, padding="same", use_bias=False))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        x = sp.SpectralNormalization(
            Conv2D(cnum * 8, kernel_size=dkernel_size, strides=1, padding="same", use_bias=False))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        classifier = Conv2D(1, kernel_size=dkernel_size, strides=1, padding="same")

        if use_sigmoid == True:
            sig = Activation("sigmoid")(x)
            out = classifier(sig)
        else:
            out = classifier(x)
        print("Discriminator Structure:")
        Discriminator = Model(input, out, name="Discriminator")
        Discriminator.summary()
        return Discriminator

    def build_model(self, input_shape, cnum=32):
        """
        Build up the PEN-Net structure(Discriminator,Generator and Combined Model)
        :param input_shape: Input Shape for the PEN-Net (channel[0:2] for Img,channel[-1] for mask)
        :param cnum: Conv channel number
        :return: Discriminator,Generator and Combined Model
        """
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
        Discriminator = self.build_discriminator(img_shape)
        Discriminator.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
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

    def Initialize_Model(self):
        """
        To prevent the "FailedPreconditionError"
        :return: Useless data
        """
        Batch_Img_Initialize = img255_normalization(
            np.ones([self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]]))
        Masked_Img_Batch_Initialize, Masks_Initialize = Centre_Mask(Batch_Img_Initialize, self.mask_size)
        return self.Generated_Img(Masked_Img_Batch_Initialize, Masks_Initialize)[0]

    def Generated_Img(self, Img, Masks):
        """
        Generate the final PEN-Net output with the input Img and Mask
        :param Img: Img array
        :param Masks: Mask array
        :return: Final Output of the PEN-Net
        """
        Masked_Imgs = Masked_Img(Img, Masks)
        Org_Predicted_Imgs = self.Generator.predict([Img2Img_with_mask(Masked_Imgs, Masks), Masks])[0]
        UnMasked_Imgs = Img * (1 - Masks)
        Output = Org_Predicted_Imgs * Masks + UnMasked_Imgs
        # plt.imshow(Org_Predicted_Imgs[0])
        # plt.figure()
        # plt.imshow(UnMasked_Imgs[0])
        # plt.figure()
        # plt.imshow((Org_Predicted_Imgs*Masks)[0])
        # plt.show()
        return Output

    def load_data_norm(self):
        """
        Load the normalized data for training and other purposes
        :return: Normalized Img data
        """
        Batch_Img = self.data_loader.load_facade()
        return img255_normalization(Batch_Img)

    def save_imgs(self, epcho, mask_size):
        """
        Save test imgs
        :param epcho: current epcho(used to generate the img name)
        :param mask_size: output's mask size
        :return: None
        """
        name_g = "generated_epoch_" + str(epcho) + ".png"
        name_decoding = "decoding_epoch_" + str(epcho) + ".png"
        name_real_out = "real_output_epoch_" + str(epcho) + ".png"
        path_g = "./generated_Imgs/" + name_g
        path_decoding = "./generated_Imgs/Decoding_Imgs/" + name_decoding
        path_realout = "./generated_Imgs/Real_Output_Imgs/" + name_real_out
        if not os.path.exists("./generated_Imgs"):  # 如果路径不存在
            os.makedirs("./generated_Imgs")
        if not os.path.exists("./generated_Imgs/Decoding_Imgs"):  # 如果路径不存在
            os.makedirs("./generated_Imgs/Decoding_Imgs")
        if not os.path.exists("./generated_Imgs/Real_Output_Imgs/"):  # 如果路径不存在
            os.makedirs("./generated_Imgs/Real_Output_Imgs/")
        Batch_Img = self.load_data_norm()
        # Masked_Img_Batch, Masks = Centre_Mask(Batch_Img, mask_size)
        Masked_Img_Batch, Masks = Random_Rectangle_Mask(Batch_Img, mask_size)
        # Real_Output=Generated_Img(Batch_Img,Masks)
        Output_Img, Masked_Img, UnMasked_Img, Img1, Img2, Img3, Img4, Img5 = self.Generator.predict(
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
        plt.imshow(normalization(self.Generated_Img(Batch_Img, Masks)[0]))
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

    def save_model(self):
        """
        Save the weights
        :return: None
        """
        if not os.path.exists("./models"):  # 如果路径不存在
            os.makedirs("./models")
        with open("./models/epoch.txt", "w") as f:
            f.write(str(self.current_epoch))

        self.Generator.save_weights('./models/Generator_weights.h5')
        self.Discriminator.save('./models/Discriminator_full.h5')
        self.Combined_Model.save_weights('./models/Combined_Model_weights.h5')

    def train(self):
        """
        Train the PEN-Net
        :return:None
        """
        print("Start_training....")
        valid = np.ones(shape=[self.batch_size, 16, 16, 1])
        fake = np.zeros(shape=[self.batch_size, 16, 16, 1])

        # follow multi-thread approach to read data
        Batch_Img = self.load_data_norm()
        for epoch in range(self.epochs + 1):
            epoch = epoch + self.previous_epoch
            self.current_epoch = epoch

            try:
                if Batch_Img == None:
                    Batch_Img = self.load_data_norm()
            except:
                pass
            load_thread = LoadingThread(eval('self.load_data_norm'))
            load_thread.start()
            # train Discriminator

            # useless when using the multi-thread

            # Batch_Img = Load_data_norm(batch_size)
            # Real_img=Load_data_norm(batch_size)

            # use a advanced model to generate mask instead of the simple centre mask one
            Masked_Img_Batch, Masks = Random_Rectangle_Mask(Batch_Img, self.mask_size)
            # Masked_Img_Batch, Masks=Centre_Mask(Batch_Img, mask_size)
            Fake_img = self.Generated_Img(Masked_Img_Batch, Masks)
            # Fake_img=Generator.predict(Img2Img_with_mask(Masked_Img_Batch,Masks), Masks)[0]
            Resized_Imgs = Pyramidal_Img_Resize(Batch_Img)

            # D_loss_real = Discriminator.train_on_batch(Batch_Img, valid)
            D_loss_real = self.Discriminator.train_on_batch(Batch_Img, valid)
            D_loss_fake = self.Discriminator.train_on_batch(Fake_img, fake)
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
            # train Generator
            G_loss = self.Combined_Model.train_on_batch([Img2Img_with_mask(Masked_Img_Batch, Masks), Masks],
                                                        [(Resized_Imgs[0] * (1 - Masks)), (Resized_Imgs[0] * Masks),
                                                         Resized_Imgs[1], Resized_Imgs[2],
                                                         Resized_Imgs[3], Resized_Imgs[4], Resized_Imgs[5],
                                                         valid])
            print("%d [D loss: %f, acc.: %.2f%%] [G L1 loss: %f , Adv loss: %f]" % (
                epoch, D_loss[0], 100 * D_loss[1], G_loss[0], G_loss[6]))

            # If at save interval => save generated image samples
            if epoch % self.save_interval == 0:
                print("Saving Models....")
                if self.Is_debug_discri:
                    self.test_discriminator_output()
                self.save_imgs(epoch, self.mask_size)
                if epoch != 0:
                    self.save_model()
            if epoch % self.mask_update_interval == 0 and epoch != 0 and self.acc_mask_size >= 0:
                self.mask_size[0] = self.mask_size[0] + self.acc_mask_size
                self.mask_size[1] = self.mask_size[1] + self.acc_mask_size
                if self.mask_size[0] >= 128:
                    self.mask_size = [128, 128]

            Batch_Img = load_thread.get_result()

    def test_discriminator_output(self):
        """
        Test for the Markovian Discriminator's output
        :return:None
        """
        print("Discriminator's output on real imgs: ", end='')
        print(np.mean(self.Discriminator.predict(self.load_data_norm())))
        print("Discriminator's output on real imgs: ", end='')
        Masked_Img_Batch, Masks = Random_Rectangle_Mask(self.load_data_norm(), self.mask_size)
        print(np.mean(self.Discriminator.predict(self.Generated_Img(Masked_Img_Batch, Masks))))

    def test_console_app(self, Is_saving=True):
        """
        Console app for testing the PEN-Net
        :param Is_saving:  whether the output should be save
        :return:None
        """
        from time import time
        name_num = 0
        print("Start test console app!\n")
        while True:
            a = input("Continue? y/n \n")
            if a == "y":
                try:
                    img_path = input("Please enter the path of the img that you are going to test:\n")
                    img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))
                    start_time = time()
                    img = scipy.misc.imresize(img, self.img_size)
                    input_img, input_mask = GenerateValidInputImg(img)
                    input_img = np.clip((input_img + 5 * input_mask), 0, 1)
                    # plt.imshow(input_mask[0,:,:,0],cmap="gray")
                    # plt.show()
                    Output_Img, Masked_Img, UnMasked_Img, Img1, Img2, Img3, Img4, Img5 = self.Generator.predict(
                        [input_img, input_mask])
                    plt.imshow(Output_Img[0])
                    plt.axis("off")
                    if Is_saving:
                        plt.pause(1)
                        if not os.path.exists("./generated_Imgs/test_app"):  # 如果路径不存在
                            os.makedirs("./generated_Imgs/test_app")
                        plt.imsave("./generated_Imgs/test_app/processed_test_" + str(name_num) + ".png",
                                   np.clip(Output_Img[0], 0, 1))
                        print("Finished! Used time:" + str(time() - start_time) + "s!\n")
                        name_num = name_num + 1
                    else:
                        plt.show()
                except:
                    print("Invalid input!")
            elif a == "n":
                print("exit")
                break
            else:
                print("Please enter y/n!\n")
                continue


def Config_GPU():
    """
    Config the GPU to prevent memory errors
    :return: None
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


if __name__ == "__main__":
    Config_GPU()
    pennet = PENNet()
    pennet.Is_Auto_Loading = False
    pennet.Is_debug_discri = True
    pennet.train()
    pennet.test_console_app()
