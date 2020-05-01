# encoding: utf-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # 加载nn中的功能函数
import torch.optim as optim  # 加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets, transforms  # 加载计算机视觉有关包
from torch.autograd import Variable

from DealDataset import DealDataset
from matplotlib import pyplot
import matplotlib.pyplot as plt
BATCH_SIZE = 512

# 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
train_dataset = DealDataset('MNIST_data/', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                           transform=transforms.ToTensor())
test_dataset = DealDataset('MNIST_data/', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                          transform=transforms.ToTensor())

# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义网络模型亦即Net 这里定义一个简单的全连接层784->10
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 10)

    def forward(self, X):
        return F.relu(self.linear1(X))


model = Model()  # 实例化全连接层
loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 10

# 以下四个列表是为了可视化（暂未实现）
losses = []
acces = []
eval_losses = []
eval_acces = []

for echo in range(num_epochs):
    start = time.time()
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    model.train()  # 将网络转化为训练模式
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        X = X.view(-1, 784)  # X:[64,1,28,28] -> [64,784]将X向量展平
        X = Variable(X)  # 包装tensor用于自动求梯度
        label = Variable(label)
        out = model(X)  # 正向传播
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数

        # 计算损失
        train_loss += float(lossvalue)
        # 计算精确度
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:" + ' ' + str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' ' + str(train_acc / len(train_loader)))
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 模型转化为评估模式
    for X, label in test_loader:
        X = X.view(-1, 784)
        X = Variable(X)
        label = Variable(label)
        testout = model(X)
        testloss = loss(testout, label)
        eval_loss += float(testloss)

        _, pred = testout.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print("testlose: " + str(eval_loss / len(test_loader)))
    print("testaccuracy:" + str(eval_acc / len(test_loader)) + '\n')


    end = time.time()

    # print("Times:%.2fs"%(end-start))
    print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))

x = range(num_epochs)
plt.figure()
plt.plot(x, eval_losses, marker='o', mec='r', mfc='w', label='testlose')
plt.plot(x, eval_acces, marker='*', ms=10, label='testaccuracy')
plt.legend()  # 让图例生效

plt.xlabel('echo')  # X轴标签
plt.figure()
plt.plot(x, losses, marker='o', mec='r', mfc='w', label='lose')
plt.plot(x, acces, marker='*', ms=10, label='accuracy')
plt.legend()  # 让图例生效

plt.xlabel('echo')  # X轴标签
plt.show()