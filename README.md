# PEN-Net   Keras-Master
## Image Inpainting task

![PEN-Net](https://github.com/qxdnfsy/PEN-Net-Keras-Img_Inpainting/blob/master/demo/demo.png?raw=true)

## Introduction
PEN-Net was rewrited into Keras now! 
which means it is easier to use !

"PEN-Net" is CVPR-2019 paper for Image Inpainting task, see reference for more details!

## Demo
PEN-Net Decoding process visualization:
 ![PEN-Net](https://github.com/qxdnfsy/PEN-Net-Keras-Img_Inpainting/blob/master/demo/decoding.png?raw=true)

Demo2:
 ![PEN-Net](https://github.com/qxdnfsy/PEN-Net-Keras-Img_Inpainting/blob/master/demo/demo2.png?raw=true)

Notice: This model have some certain unknown issue need to be fixed as it can not produce high quality visual realistic detail, I am trying to fix it currently. 
If you know waht's the problem, feel free to do the pull request!

## Requirements
* scipy==1.1.0
* pillow
* numpy
* tensorflow-gpu
* keras
* matplotlib

* (programed in python 3.6)

## Test
1. put correct model under the "/models" folder
2. init the PENNet class and call the class function "test_app_console" for testing
3. The purely white pix of the input img will be classify as the "mask" automatically
4. the output image will be in the "/generated_Imgs/test_app"

## Train
1. init the PENNet class and the relevant configs (batch_size,dataset_path,etc.)
2. select a proper type(or way) of how dataloader works
3. just do the training by running the program! The model and epcho of saving will be saved directly by PENNet class automatically

## Reference 
1. [Arxiv Paper](https://arxiv.org/abs/1904.07475)
2. [Pytorch-version](https://github.com/1900zyh/PEN-Net-for-Inpainting)

## License
Licensed under an MIT license.
