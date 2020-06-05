# 导入库
import argparse
import os

from Dataset import Dataset

from torch.autograd import Variable
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# 设置数据集路径




# helper function for data visualization
from unet.unet_model import UNet


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.pause(0.1)
    # plt.show()

#### Visualize resulted augmented images and masks

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(height=320, width=320, always_apply=True),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),

    ]
    return albu.Compose(train_transform)

def get_test_augmentation():
    train_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)

# augmented_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     augmentation=get_training_augmentation(),
# )
#
# # same image with different random transforms
# for i in range(1):
#     image, mask = augmented_dataset[1]
#     print(np.min(image),np.max(image))
#     print(mask)
#     visualize(image=image, mask=mask[0,:,:])






def get_args():
      parser = argparse.ArgumentParser()
      parser.add_argument('--n_classes', type=int, default=1, help='number of epochs to train for')
      parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
      parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
      parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for Critic, default=0.01')
      parser.add_argument('--optim', type=str, default='adam', help='Which optimizer to use, default is adam')
      parser.add_argument('-m', '--load', dest='load', default=None, help='load model')
      opt = parser.parse_args()
      return opt


DATA_DIR = './data/'  # 根据自己的路径来设置

x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')
if __name__ == "__main__":
    opt = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_num = [17] if opt.n_classes == 1 else ([17, 21] if opt.n_classes == 2 else [i for i in range(32)])
    print(class_num)
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        n_classes=opt.n_classes,
        class_num = [17] if opt.n_classes==1 else ([17,21] if  opt.n_classes==2 else [i for i in range(32)])
        # class_num = [17,21]
        # class_num=[i for i in range(32)]
    )


    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    net = UNet(n_channels=3, n_classes=opt.n_classes)
    net.cuda()
    if opt.optim == 'rms':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=1e-8)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    if net.n_classes > 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    if opt.load is not None:
        model_dir = './checkpoints/epoch_4.pth'  # 模型存放地
        # model_dir = './model/all.pth'  # 模型存放地
        if not os.path.exists(model_dir):
            print('model not found')
        else:
            print('load exist model')
            net.load_state_dict(torch.load(model_dir))

    # 40  0.01
    # 40 0.005
    # 20 0.001
    # device = 'cuda'



    for epoch in range(opt.epoch):

        net.train()
        epoch_loss = 0

        for data in train_loader:
            images, labels = data
            images = images.permute(0, 3, 1, 2)
            # print('qian')
            # print(images[0,0,0])
            # print(labels[0, 0, 0])
            images = Variable(images.to(device=device, dtype=torch.float32))
            labels = Variable(labels.to(device=device, dtype=torch.float32))
            images = images / 255.0
            # print('hou')
            # print(images[0, 0, 0])
            # print(images)
            # print(labels)
            pred = net(images)

            # wrong to use loss = criterion(pred.view(-1), labels.view(-1))
            loss = criterion(pred, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch：', epoch, 'epoch_loss: ', epoch_loss / len(train_loader))
        if not os.path.exists('./checkpoints/'):
            os.makedirs('./checkpoints/')
        torch.save(net.state_dict(), './checkpoints/epoch_%d.pth' % epoch)









