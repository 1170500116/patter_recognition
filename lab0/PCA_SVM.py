import numpy as np
import struct
import matplotlib.pyplot as plt
import os
##加载svm模型

from sklearn import svm
###用于做数据预处理
from sklearn import preprocessing
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from DealDataset import DealDataset
def mk_dataset(size,n):
    """makes a dataset of size "size", and returns that datasets images and targets
    This is used to make the dataset that will be stored by a model and used in
    experimenting with different stored dataset sizes
    """


    train_img = [img[i].numpy().flatten() for i in indx[:size]]
    # print(type(train_img))
    train_img = np.array(train_img)
    train_target = [target[i] for i in indx[:size]]
    train_target = np.array(train_target)
    # b = []
    # for i in train_target:
    #     if i not in b:
    #         b.append(i)
    # print(b)

    test_img = [img[i].numpy().flatten() for i in indx[size:n]]
    test_img = np.array(test_img)
    test_target = [target[i] for i in indx[size:n]]
    test_target = np.array(test_target)
    # bb = []
    # for i in test_target:
    #     if i not in bb:
    #         bb.append(i)
    #
    # print(bb)


    return train_img, train_target,test_img,test_target

trainDataset = DealDataset('MNIST_data/', "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",transform=transforms.ToTensor())
testDataset = DealDataset('MNIST_data/', "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",transform=transforms.ToTensor())





img,target =  trainDataset.__getall__()
img1,target1 =  testDataset.__getall__()
for i in range(len(img1)):
    img.append(img1[i])
    target.append(target1[i])
print(len(target))
print("?????????????")

indx = np.random.choice(len(target), 70000, replace=False)
train_img, train_target,test_img,test_target = mk_dataset(60000,70000)
print(len(target))
print("?????????????")

start = time.time()
scaler = StandardScaler()

# 注意fit时我们仅仅对训练集进行。
scaler.fit(train_img)
# 再对训练集和测试集均进行transform
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
pca = PCA(n_components=0.8)

# 再次注意，fit只读训练数据fit。
# 实际上这里是要学习特征空间
pca.fit(train_img)

# 之后对训练集和测试集进行数据压缩
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print(train_img[0].shape)
end = time.time()
print("Times:%.2fs" % (end - start))
print(int(int(end - start) / 60) * 60)
print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))
varience_num = pca.n_components_
print(varience_num)





# 标准化

X_train =train_img
y_train = train_target
print(len(train_img[0]))
print("?????????????")
# 模型训练
start = time.time()
# model_svc = svm.SVC()
model_svc = svm.SVC(C=8.0, kernel='rbf', gamma=1.0/784,cache_size=8000,probability=False)
model_svc.fit(X_train, y_train)
end = time.time()
print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))

# 评分并预测
x_test = test_img
y_pred = test_target
print(model_svc.score(x_test, y_pred))
y = model_svc.predict(x_test)
print( "每个类的支持向量的个数: ", model_svc.n_support_)
#
# print "支持向量: ", clf.support_vectors_
# print "正类和负类的支持向量的索引: ", clf.support_



# 模型训练
start = time.time()
# model_svc = svm.SVC()
model_svc = svm.SVC(C=80.0, kernel='rbf', gamma=1.0/784,cache_size=8000,probability=False)
model_svc.fit(X_train, y_train)
end = time.time()
print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))

# 评分并预测
x_test = test_img
y_pred = test_target
print(model_svc.score(x_test, y_pred))
y = model_svc.predict(x_test)
print( "每个类的支持向量的个数: ", model_svc.n_support_)

# 模型训练
start = time.time()
# model_svc = svm.SVC()
model_svc = svm.SVC(C=800.0, kernel='rbf', gamma=1.0/784,cache_size=8000,probability=False)
model_svc.fit(X_train, y_train)
end = time.time()
print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))

# 评分并预测
x_test = test_img
y_pred = test_target
print(model_svc.score(x_test, y_pred))
y = model_svc.predict(x_test)
print( "每个类的支持向量的个数: ", model_svc.n_support_)


# # 模型训练
# start = time.time()
# # model_svc = svm.SVC()
# model_svc = svm.SVC(C=80.0, kernel='rbf', gamma=1.0/784,cache_size=8000,probability=False)
# model_svc.fit(X_train, y_train)
# end = time.time()
# print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))
#
# # 评分并预测
# x_test = test_img
# y_pred = test_target
# print(model_svc.score(x_test, y_pred))
# y = model_svc.predict(x_test)
# print( "每个类的支持向量的个数: ", model_svc.n_support_)






