import os
import random
import shutil

# 数据集路径
dataset_path = './data'
images_path = './camvid/images'
labels_path   = './camvid/labels'

images_name = os.listdir(images_path)
images_num  = len(images_name)
alpha  = int( images_num  * 0.8 )
beta   = int( images_num  * 0.2 )

print(images_num)

random.shuffle(images_name)

train_list = images_name[0:alpha]
test_list = images_name[alpha:images_num]


# 确认分割正确
print('train list: ',len(train_list))
print('test list: ',len(test_list))
print('total num: ',len(test_list)+len(train_list))

# 创建train,valid和test的文件夹
train_images_path = os.path.join(dataset_path,'train_images')
train_labels_path  = os.path.join(dataset_path,'train_labels')
if os.path.exists(train_images_path)==False:
    os.mkdir(train_images_path )
if os.path.exists(train_labels_path)==False:
    os.mkdir(train_labels_path)


test_images_path = os.path.join(dataset_path,'test_images')
test_labels_path  = os.path.join(dataset_path,'test_labels')
if os.path.exists(test_images_path)==False:
    os.mkdir(test_images_path )
if os.path.exists(test_labels_path)==False:
    os.mkdir(test_labels_path)
print(os.getcwd())
 # 拷贝影像到指定目录
for image in train_list:
    shutil.copy(os.path.join(images_path,image), os.path.join(train_images_path,image))
    shutil.copy(os.path.join(labels_path,image), os.path.join(train_labels_path,image))


for image in test_list:
    shutil.copy(os.path.join(images_path,image), os.path.join(test_images_path,image))
    shutil.copy(os.path.join(labels_path,image), os.path.join(test_labels_path,image))
