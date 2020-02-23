from core.utils import *
import numpy as np


def Random_Rectangle_Mask(img, shape=[128, 128], Is_Centred=False, Is_Random_size=True, Is_square=True,
                          Maximum_smaller_scale=1.5):
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
