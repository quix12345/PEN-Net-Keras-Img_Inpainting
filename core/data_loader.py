import os
import scipy.misc
from core.utils import *


class data_loader:
    def __init__(self, dataset_path,batch_size):
        self.dataset_path = dataset_path + "/"
        self.batch_size=batch_size

    def Load_data(self, batch_size=-1, img_size=[96, 96]):
        if batch_size==-1:
            batch_size=self.batch_size
        all_image_path = [self.dataset_path + x for x in os.listdir(self.dataset_path)]
        index = np.random.randint(batch_size + 1, len(all_image_path))
        batch_img_path = all_image_path[index - batch_size:index]
        img_train_data = np.zeros([batch_size, img_size[0], img_size[1], 3])
        i = 0
        for img_path in batch_img_path:
            img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))
            img = scipy.misc.imresize(img, img_size)
            img_train_data[i] = img
            i = i + 1

        return np.array(img_train_data, dtype=int)

    def load_data_with_auto_enhancement(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        batch_imgs = []
        batch_path = [self.dataset_path + x for x in os.listdir(self.dataset_path)]
        i = 0
        while i < batch_size:
            index = np.random.randint(0, len(batch_path) - 1)
            img_path = batch_path[index]
            img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))
            x, y, _ = img.shape
            img = Resize_Img(img)
            brighten_factor = np.random.uniform(0.8, 1.2)
            mode = np.random.randint(0, 2)
            if mode == 1:
                mode = 2
            else:
                mode = 0
            img = img_brighten(img_reversed(img, mode), brighten_factor)
            new_img = Random_Rectangle_Img(img)
            batch_imgs.append(new_img)
            i = i + 1
        return np.array(batch_imgs, dtype=int)

    def load_facade(self, batch_size=-1, suffix='jpg', path_base="F:/AI_Datasets/CMP_facade_DB_base/base/",
                    path_extended='F:/AI_Datasets/CMP_facade_DB_extended/extended/'):
        """
        Load facade data
        :param batch_size:
        :param suffix:
        :param path_base:
        :param path_extended:
        :return: loaded data
        """
        if batch_size == -1:
            batch_size = self.batch_size
        batch_imgs = []
        choice = np.random.randint(0, 2)
        if choice == 0:
            path = path_base
        else:
            path = path_extended
        batch_path = [path + x for x in os.listdir(path)]
        i = 0
        while i < batch_size:
            index = np.random.randint(0, len(batch_path) - 1)
            img_path = batch_path[index]
            if img_path[-3:] == suffix:
                img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))
                x, y, _ = img.shape
                img = Resize_Img(img)
                brighten_factor = np.random.uniform(0.8, 1.2)
                mode = np.random.randint(0, 2)
                if mode == 1:
                    mode = 2
                else:
                    mode = 0
                img = img_brighten(img_reversed(img, mode), brighten_factor)
                new_img = Random_Rectangle_Img(img)
                batch_imgs.append(new_img)
                i = i + 1
        return np.array(batch_imgs, dtype=int)
