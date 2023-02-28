import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder
import numpy as np
import copy
from data_process import count_data

np.random.seed(6)
#random.seed(2)
def build_dataset(dataset,num_meta):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(40), # 40
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=transform_test)
        img_num_list = [num_meta] * 10
        num_classes = 10

    if dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100('../data', train=False, transform=transform_test)
        img_num_list = [num_meta] * 100
        num_classes = 100
    
    ################################# prove the availability #######################################
    if dataset == 'bdd100k':
        train_dataset = ImageFolder(root="/home/chengru/github/Longtail_DA-master/bdd100k/train/",transform=transform_train)
        # class starts from 0
        # train_dataset[0][0] is image, train_dataset[0][1] is the label for the image
        test_dataset = ImageFolder(root="/home/chengru/github/Longtail_DA-master/bdd100k/val/",transform=transform_test)
        _, num_each = count_data(path="/home/chengru/github/Longtail_DA-master/bdd100k/train_day.json")
        num_classes = 10
        img_num_list = [num_meta] * 10
        for i in range(len(num_each)):
            if num_each[i] < num_meta:
                img_num_list[i] = num_each[i]

        # train_data, train_data_meta = curation(copy.deepcopy(train_dataset), img_num_list, num_classes, keep_rate)
        # return train_data_meta,train_data,test_dataset 


    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]

    idx_to_meta = []
    idx_to_train = []
    print(img_num_list)

    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])

    train_data = copy.deepcopy(train_dataset)
    train_data_meta = copy.deepcopy(train_dataset)

    if dataset == "bdd100k":
        train_data_meta.samples = np.delete(train_dataset.samples,idx_to_train,axis=0)
        train_data_meta.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)
        train_data.samples = np.delete(train_dataset.samples, idx_to_meta, axis=0)
        train_data.targets = np.delete(train_dataset.targets, idx_to_meta, axis=0)
    else:
        train_data_meta.data = np.delete(train_dataset.data,idx_to_train,axis=0)
        train_data_meta.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)
        train_data.data = np.delete(train_dataset.data, idx_to_meta, axis=0)
        train_data.targets = np.delete(train_dataset.targets, idx_to_meta, axis=0)

    return train_data_meta,train_data,test_dataset,train_dataset


def get_img_num_per_cls(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


# This function is used to generate imbalanced test set
'''
def get_img_num_per_cls_test(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (10000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (10000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls
'''
