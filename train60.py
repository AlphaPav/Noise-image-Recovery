from model import ConvColorNet

from ImageHelper import ImageHelper
import torch
import torchvision
import torch.nn as nn
import config
import os
import numpy as np
import cv2


def load_last_ckpt(mode):
    max=-1
    for file in os.listdir('./log/'+mode):
        temp=int(file.split('_')[1].split('.')[0])
        if temp>max:
            max=temp
    return max


def calc_loss(outputs, true_images, batch_size):
    err = 0
    for i in range(0,batch_size):
        output= outputs[i]
        true_image= true_images[i]
        for j in range(0,3):
            # print(color40[j].shape) # torch.Size([32, 32])
            # print(color40[j].view(1,-1).squeeze())
            err+= torch.norm(output[j].view(1,-1).squeeze()  - true_image[j].view(1,-1).squeeze())
            # print("err",err)
    return err

def train_color_60(pretrained=False):

    # 定义损失函数和优化器
    device =config.device
    model = ConvColorNet(3).to(device)


    if pretrained == False:
        ckpt_id = 0
    else:
        ckpt_id = load_last_ckpt('color60')
        print('load pretrained model at ckpt_{}'.format(ckpt_id))
        model.load_state_dict(torch.load('./log/color60/' + 'ckpt_' + str(ckpt_id) + '.pkl'))
        ckpt_id += 1

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    imageHelper = ImageHelper()
    imageHelper.load_data()

    total_step = len(imageHelper.train_loader)
    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(imageHelper.train_loader):
            # 注意模型在GPU中，数据也要搬到GPU中
            # 一组batch
            true_images = images.to(device) #  true image
            noise_images= imageHelper.get_noise_batch(true_images,0.6,config.batch_size).to(device)
            # labels = labels.to(device)
            # print(labels)
            # 前向传播
            outputs = model(noise_images)
            loss= calc_loss(outputs, true_images, config.batch_size)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item()))
                tempimage = outputs[0].detach().cpu().numpy()  # numpy
                tempimage= np.transpose(tempimage, (1, 2, 0))
                cv2.imwrite('./color60/epoch'+ str(epoch+1)+'_step'+ str(i+1)+'_output.png', tempimage*255)
                trueimage = true_images[0].cpu().numpy()
                trueimage = np.transpose(trueimage, (1, 2, 0))
                cv2.imwrite('./color60/epoch' + str(epoch + 1) + '_step' + str(i + 1) + '_true.png', trueimage * 255)
                noiseimage = noise_images[0].cpu().numpy()
                noiseimage = np.transpose(noiseimage, (1, 2, 0))
                cv2.imwrite('./color60/epoch' + str(epoch + 1) + '_step' + str(i + 1) + '_noise.png', noiseimage * 255)

        print('save ckpt\n')
        torch.save(model.state_dict(), './log/color60/' + 'ckpt_' + str(epoch + ckpt_id) + '.pkl')
        # val_color_60(epoch + ckpt_id)

def val_color_60():
    device = config.device
    model = ConvColorNet(3).to(device)

    ckpt_id = load_last_ckpt('color60')
    print('load pretrained model at ckpt_{}'.format(ckpt_id))
    model.load_state_dict(torch.load('./log/color60/' + 'ckpt_' + str(ckpt_id) + '.pkl'))

    imageHelper = ImageHelper()
    imageHelper.load_data()

    total_step = len(imageHelper.test_loader) # 500
    for i, (images, labels) in enumerate(imageHelper.test_loader):
        true_images = images.to(device)  # true image
        noise_images = imageHelper.get_noise_batch(true_images, 0.6, config.batch_size).to(device)
        outputs = model(noise_images)
        loss = calc_loss(outputs, true_images, config.batch_size)

        if (i + 1) % 50 == 0:
            print('Step [{}/{}], Loss: {:.4f}'
                  .format( i + 1, total_step, loss.item()))
            tempimage = outputs[0].detach().cpu().numpy()  # numpy
            tempimage = np.transpose(tempimage, (1, 2, 0))
            cv2.imwrite('./output/color60/val'  + '_step' + str(i + 1) + '_output.png',
                        tempimage * 255)
            trueimage = true_images[0].cpu().numpy()
            trueimage = np.transpose(trueimage, (1, 2, 0))
            cv2.imwrite('./output/color60/val' + '_step' + str(i + 1) + '_true.png', trueimage * 255)
            noiseimage = noise_images[0].cpu().numpy()
            noiseimage = np.transpose(noiseimage, (1, 2, 0))
            cv2.imwrite('./output/color60/val' + '_step' + str(i + 1) + '_noise.png',
                        noiseimage * 255)


if __name__ == '__main__':
    # train_color_60(pretrained=True)
    val_color_60()
    # test_color_60("C.png")