# -*- coding:utf-8 -*-



'''
系统环境: Windows10
Python版本: 3.7
PyTorch版本: 1.3.1
cuda: 80
'''
import torch
import torch.nn.functional as F  # 激励函数的库
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
# 定义全局变量
from DealDataset import  CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset


n_epochs = 20  # epoch 的数目
batch_size = 64  # 决定每次读取多少图片


# 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor

transform = transforms.Compose([
    # transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  # 先四周填充0，再把图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

root = './cifar-10-batches-py'
train_dataset = CIFAR10(root, transform=transform, target_transform=None, train=True)
test_dataset = CIFAR10(root, transform=transform, target_transform=None, train=False)
# train_dataset = DealDataset(data_train, label_train,
#                            transform=transform)
# test_dataset = DealDataset(data_test, label_test,
#                            transform=transform)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

class AlexNet(nn.Module):  # 定义网络，推荐使用Sequential，结构清晰
    def __init__(self,num_classes = 10):
        super(AlexNet, self).__init__()

        self.conv = torch.nn.Sequential(  # input_size = 32*32*3
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # output_size = 16*16*96

            # input_size = 16*16*96
            torch.nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2) , # output_size = 8*8*256
            # input_size =  8*8*256
            torch.nn.Conv2d(192, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            torch.nn.ReLU(),  # output_size =  8*8*384

            # input_size = 8*8*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(),  # output_size = 8*8*384


            # input_size = 8*8*384
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # output_size = 4*4*256

        )


        # 网络前向传播过程
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, num_classes)
        )

    def forward(self, x):  # 正向传播过程
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out
# 定义网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积层
        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，6个卷积核，核大小5*5
            # 经过该层图像大小变为32-5+1，28*28
            # 经2*2最大池化，图像变为14*14
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 卷积层2，6输入通道，16个卷积核，核大小5*5
            # 经过该层图像变为14-5+1，10*10
            # 经2*2最大池化，图像变为5*5
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            # 16个feature，每个feature 5*5
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


train_losses=[]
valid_losses=[]
# 训练神经网络
def train():
    # 定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # 开始训练
    for epoch in range(n_epochs):
        startTick = time.clock()
        model.train()
        train_loss = 0.0
        i = 0
        for data, target in train_loader:
            # print(target)
            data, target = data.to(device), target.to(device)
            # print(type(data))
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
            # print(train_loss)
            train_losses.append(train_loss)
            if(i*batch_size%10000<batch_size):
                print(i)
            i += 1
        train_loss = train_loss / len(train_loader.dataset)
        # 每遍历一遍数据集，测试一下准确率
        test_loss,Accuracy = test(epoch)
        writer.add_scalars('Adam', {'train_loss': train_loss,'test_loss': test_loss}, global_step=epoch)
        print('Epoch:  {}  \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            test_loss))
        print('Accuracy of the network on the test images: %d %% ' % (
                Accuracy))
        writer.add_scalar('Accuracy', Accuracy, global_step=epoch)


        timeSpan = time.clock() - startTick
        print("耗时%dS" % (timeSpan))






# 在数据集上测试神经网络
def test(epoch):
    correct = 0
    total = 0
    model.eval()
    test_loss = 0.0
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_loader.dataset)


    return test_loss,100 * correct / total



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 声明感知器网络
    print(device)

    writer = SummaryWriter('runs/scalar_example')
    model = AlexNet().to(device)
    print(model)
    # 保存
    train()
    writer.close()
