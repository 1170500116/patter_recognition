import heapq
import time
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader, TensorDataset, dataloader
from torchvision import transforms

from DealDataset import DealDataset
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import numpy as np

from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def skl_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use in classication
    test_data: the data/targets used to test the classifier
    stored_data: the data/targets used to classify the test_data
    """

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(stored_data, stored_target)

    y_pred = classifier.predict(test_data)

    print(classification_report(test_target, y_pred))



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


def cos_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use for voting
    test_data: a set of unobserved images to classify
    test_target: the labels for the test_data (for calculating accuracy)
    stored_data: the images already observed and available to the model
    stored_target: labels for stored_data
    """

    # find similarity for every point in test_data between every other point in stored_data
    cosim = cosine_similarity(test_data, stored_data)

    # get indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k + 1), range(len(i)), i.take)) for i in cosim]
    # convert indices to numbers
    top = [[stored_target[j] for j in i[:k]] for i in top]

    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    # print table giving classifier accuracy using test_target
    print(classification_report(test_target, pred))





if __name__=="__main__":
    trainDataset = DealDataset('MNIST_data/', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                               transform=transforms.ToTensor())
    testDataset = DealDataset('MNIST_data/', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                              transform=transforms.ToTensor())


    img, target = trainDataset.__getall__()
    img1, target1 = testDataset.__getall__()
    for i in range(len(img1)):
        img.append(img1[i])
        target.append(target1[i])
    print(len(target))

    indx = np.random.choice(len(target), 70000, replace=False)
    train_img, train_target, test_img, test_target = mk_dataset(60000, 70000)
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
    start = time.time()
    skl_knn(5, test_img, test_target, train_img, train_target)
    end = time.time()
    print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))
    start = time.time()
    skl_knn(30, test_img, test_target, train_img, train_target)
    end = time.time()
    print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))
    start = time.time()
    cos_knn(5, test_img, test_target, train_img, train_target)
    end = time.time()
    # print("Times:%.2fs"%(end-start))
    print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))
    # print("Times:%.2fs"%(end-start))
    start = time.time()
    cos_knn(30, test_img, test_target, train_img, train_target)
    end = time.time()
    # print("Times:%.2fs"%(end-start))
    print("Times:%d min %d s" % (int(end - start) / 60, int(end - start) - int(int(end - start) / 60) * 60))

    # print(trainDataset.__len__())
    # print(testDataset.__len__())
    # skl_knn(5, testDataset.train_set, testDataset.train_labels,trainDataset.train_set, trainDataset.train_labels)
    # # 训练数据和测试数据的装载
    # train_loader = dataloader.DataLoader(
    #     dataset=trainDataset,
    #     batch_size=100, # 一个批次可以认为是一个包，每个包中含有100张图片
    #     shuffle=False,
    # )
    # test_loader = dataloader.DataLoader(
    #     dataset=testDataset,
    #     batch_size=100,
    #     shuffle=False,
    # )