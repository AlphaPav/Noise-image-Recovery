
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import logging
import  config
logger = logging.getLogger("logger")
import copy
import random
import os
import cv2



class ImageHelper():
    def load_data(self):
        logger.info('Loading data')
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        ## data load
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        cifar10Path= './data'
        self.train_set = datasets.CIFAR10(cifar10Path, train=True, download=True,
                                              transform=transform_train)

        self.test_set = datasets.CIFAR10(cifar10Path, train=False, transform=transform_test)

        self.train_loader = DataLoader(dataset=self.train_set,
                         batch_size=config.batch_size,
                         shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_set,
                                       batch_size=config.batch_size,
                                       shuffle=True)

    def get_noise_batch(self, images, noise_percent, batch_size):
        newimages= torch.empty(images.shape)
        # print(images.shape,newimages.shape)
        for idx in range(0,batch_size):
            image = images[idx].cpu().numpy() # numpy
            image = np.transpose(image, (1, 2, 0))
            noise_image= self.gen_noise_img(image,noise_percent)
            noise_image = np.transpose(noise_image, (2, 0, 1))
            noise_image = torch.from_numpy(noise_image) #tenosr
            newimages[idx]= noise_image

        # print(images)
        # print(newimages)
        return newimages


    def rgb2gray(self,rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def get_gray_noise_batch(self, images, noise_percent, batch_size):
        noiseimages = torch.empty(images.shape[0],1,images.shape[2],images.shape[3])
        grayimges= torch.empty(images.shape[0],1,images.shape[2],images.shape[3])
        # print(images.shape,newimages.shape)
        for idx in range(0, batch_size):
            image = images[idx].cpu().numpy()  # numpy
            image = np.transpose(image, (1, 2, 0))
            # plt.imshow(image)
            # plt.show()
            gray = self.rgb2gray(image)
            gray_image= np.empty((1,gray.shape[0],gray.shape[1]), dtype=np.float)
            gray_image[0]= copy.deepcopy(gray)
            gray_image= torch.from_numpy(gray_image)

            gray = self.gen_gray_noise_img(gray, noise_percent)
            noise_image = np.empty((1,gray.shape[0],gray.shape[1]), dtype=np.float)
            noise_image[0]= gray
            noise_image = torch.from_numpy(noise_image)  # tenosr
            # print(noise_image.shape)
            noiseimages[idx] = noise_image
            grayimges[idx] = gray_image
        return grayimges, noiseimages

    def gen_gray_noise_img(self,image,noise_percent):

        img = copy.deepcopy(image)
        height, width = img.shape[:2]
        real_num = int(width * (1 - noise_percent))  # 每行真实像素的个数
        for i in range(0, height):  # 每一行
            mask = np.zeros(width)
            mask[:real_num] = 1
            random.shuffle(mask)
            for j in range(0, width):  # 列
                img[i][j] *= mask[j]

        # plt.imshow(img)
        # plt.show()
        return img

    def gen_noise_img(self,image,noise_percent):
        # plt.imshow(image)
        # plt.show()
        img = copy.deepcopy(image)
        height, width = img.shape[:2]
        real_num= int(width*(1-noise_percent)) # 每行真实像素的个数
        for i in range(0,height): # 每一行
            for k in range(0, 3):  # 每个通道
                mask = np.zeros(width)
                mask[:real_num] = 1
                random.shuffle(mask)
                for j in range(0,width): # 列
                        img[i][j][k] *=mask[j]

        # plt.imshow(img)
        # plt.show()
        # cv2.imwrite("cifa10_ori.png", image)
        # cv2.imwrite("cifa10_noise.png", img)

        return img

    def read_image(self,srcPath):
        img = cv2.imread(srcPath, -1)

        if img.all() == None:
            print( "Error: could not load image")
            "Error: could not load image"
            os._exit(0)
        return img/255

if __name__ == '__main__':
    imageHelper= ImageHelper()
    # imageHelper.load_data()

    noise_image= imageHelper.read_image("ex_noise.jpg")  # numpy
    noise_image= noise_image[:,:,:3]
    print(noise_image.shape)
    cv2.imwrite("ex_noise3.png", noise_image*255)


    # noise_image = torch.from_numpy(noise_image)  # tenosr
    # noise_images = torch.empty(1, 1, noise_image.shape[0], noise_image.shape[1])
    # noise_images[0][0] = noise_image
    # noise_images = noise_images
    # print(noise_images.shape)

    # gray = imageHelper.rgb2gray(noise_image)
    # print(gray.shape)
    # gray = imageHelper.gen_gray_noise_img(gray, 0.1)
    # cv2.imwrite("gray.png", gray*255)


    # for i, (images, labels) in enumerate(imageHelper.train_loader):
    #     true_images = images  # true image
    #     noise_images = imageHelper.get_gray_noise_batch(true_images, 0.4, config.batch_size)
    #     print(noise_images.shape)
    #     print(noise_images)
    #
    #     break



