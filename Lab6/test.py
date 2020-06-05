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
from unet.unet_model import UNet

DATA_DIR = './data/'  # 根据自己的路径来设置

x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.pause(0.5)
    plt.clf()
    plt.cla()


def save_img(t,**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.pause(0.1)
    # plt.show()
    if not os.path.exists('./ans/'):
        os.makedirs('./ans/')
    save_path= './ans/'
    plt.savefig(os.path.join(save_path, '{}.png'.format(t))  )

    plt.close()


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
    opt = parser.parse_args()
    return opt


DATA_DIR = './data/'  # 根据自己的路径来设置

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')
if __name__ == "__main__":
    plt.close('all')
    opt = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_num = [17] if opt.n_classes == 1 else ([17, 21] if opt.n_classes == 2 else [i for i in range(32)])
    print(class_num)
    net = UNet(n_channels=3, n_classes=opt.n_classes)
    net.cuda()
    net.eval()

    # optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-8)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.5, 0.999))
    if net.n_classes > 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 40  0.01
    # 40 0.005
    # 20 0.001
    # device = 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_dir = './checkpoints/epoch_4.pth'  # 模型存放地
    if opt.n_classes ==1:
        model_dir = './model/road.pth'  # 模型存放地
    elif opt.n_classes ==2:
        model_dir = './model/sky_road.pth'  # 模型存放地
    else :
        model_dir = './model/all.pth'  # 模型存放地
    print('load model',model_dir)
    net.load_state_dict(torch.load(model_dir))

    test_dataset_noaug = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_test_augmentation(),
        n_classes=opt.n_classes,
        class_num=[17] if opt.n_classes == 1 else ([17, 21] if opt.n_classes == 2 else [i for i in range(32)])
    )

    test_loader = DataLoader(test_dataset_noaug, batch_size=1, shuffle=False)
    # i = 0
    plt.figure(figsize=(16, 5))
    with torch.no_grad():
        for data in test_loader:
            image, mask = data
            show_image = image.squeeze(0)
            with torch.no_grad():
                image = image.permute(0, 3, 1, 2)
                image = Variable(image.to(device=device, dtype=torch.float32))
                mask = Variable(mask.to(device=device, dtype=torch.float32))
                image = image / 255.

                image = image.to()
                # print(image.shape)

                pred = net(image.cuda())

                pred = pred + 0.5
                pred = Variable(pred.to(device=device, dtype=torch.int32))
                pred = Variable(pred.to(device=device, dtype=torch.float32))

                pred = pred.cpu()
                mask = mask.cpu()

            # pred = pred > 0.5

            plt.ion()
            # save_img(i, image=show_image, GT=mask[0, 0, :, :], Pred=pred[0, 0, :, :])
            visualize(image=show_image, GT=mask[0, 0, :, :], Pred=pred[0, 0, :, :])
            # i += 1
            # plt.show()
    plt.close()

