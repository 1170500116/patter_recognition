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
from DealDataset import DealDataset
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter
n_epochs = 50  # epoch 的数目
batch_size = 50  # 决定每次读取多少图片


# 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
train_dataset = DealDataset('MNIST_data/', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                           transform=transforms.ToTensor())
test_dataset = DealDataset('MNIST_data/', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                          transform=transforms.ToTensor())


#拆分数据集

#添加验证集，让模型自动判断是否过拟合
valid_size = 0.2
transform = transforms.ToTensor()


num_workers = 0
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx,valid_idx = indices[split:],indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,
                            sampler = train_sampler)
valid_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,
                            sampler = valid_sampler)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 建立一个三层感知机网络
class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP, self).__init__()  #
        self.dropout = nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(784, 128)  # 784->128
        self.fc2 = torch.nn.Linear(128, 10)  # 128->10

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))  # 使用 relu 激活函数
        dout = self.dropout(dout)
        dout = F.softmax(self.fc2(dout))  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout

train_losses=[]
valid_losses=[]
# 训练神经网络
def train():
    # 定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    valid_loss_min = np.Inf
    # 开始训练
    for epoch in range(n_epochs):
        startTick = time.clock()
        model.train()
        train_loss = 0.0
        valid_loss = 0.0
        for data, target in train_loader:
            # print(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
            train_losses.append(train_loss)

        # 计算检验集的损失，这里不需要反向传播
        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = lossfunc(output, target)
            valid_loss += loss.item() * data.size(0)
            valid_losses.append(valid_loss)
        train_loss = train_loss / len(train_idx)
        valid_loss = valid_loss / len(valid_idx)
        writer.add_scalars('SGD',{ 'train_loss':train_loss,'valid_loss': valid_loss}, global_step=epoch)
        # writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
        print('Epoch:  {}  \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss))
        if valid_loss <= valid_loss_min:  # 保存模型
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
        # 每遍历一遍数据集，测试一下准确率
        test(epoch)
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
    print('Accuracy of the network on the test images: %d %%,test_loss = %.6f ' % (
            100 * correct / total,test_loss))
    writer.add_scalar('Accuracy', 100.0 * correct / total, global_step=epoch)
    writer.add_scalar('test_loss', 100.0 * correct / total, global_step=epoch)

    return 100.0 * correct / total



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 声明感知器网络
    print(device)
    writer = SummaryWriter('runs/scalar_example')
    model = MLP().to(device)
    # 保存
    train()
    writer.close()
