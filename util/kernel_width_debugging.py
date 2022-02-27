import cv2
import numpy as np
import glob
import time
import random
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.utils.data as data

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    # max number of images
    n_image = 20000

    dataset_root_dict = {'CIFAR10': r"C:\Users\Kim\Desktop\dataset\CIFAR10-ImageNet_codetest_32x32\trainA",
                         'ImageNet': r"C:\Users\Kim\Desktop\dataset\CIFAR10-ImageNet_codetest_32x32\trainA"}

    dataset_name = 'ImageNet'
    assert dataset_name in dataset_root_dict.keys(), 'dataset name is not correct. choose between "CIFAR10", "ImageNet"'
    fileroot = dataset_root_dict[dataset_name]
    print("fileroot is" , fileroot)

    filenames = glob.glob(fileroot)
    filenames.sort()
    random.shuffle(filenames) # if random shuffle

    # load and preporcess
    if dataset_name == 'CIFAR10':
        transform_train = T.Compose([
            T.Resize(32),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616],
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),

        ])
    if dataset_name == 'ImageNet':
        transform_train = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ConvertImageDtype(torch.float)
        ])

        train_dataset = ImageFolder(root=fileroot, transform=transform_train) # 50000

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)


    # helper functions / classes


    def filewritetest():
        with open("./dist_list.txt", 'w') as f :
            f.write('test')


    def calc_distance(dataloader):
        total_dist_list= []
        total_image_list = []
        dataiter = iter(dataloader)

        i = 0
        # compute distance
        for i in range(n_image):
            start = time.time()
            ## load new image
            im = dataiter.next()
#            print("im.shape", im.shape)
            im = im.reshape([1, -1])

            total_image_list.append(im)

            ## compute 2d distances between previous load images
            load_image_before = i
            newly_add_dist_list = []
            for i in range(load_image_before):
                previous_image = total_image_list[i]
                temp_l2_dist = np.linalg.norm(im - previous_image)
                newly_add_dist_list.append(temp_l2_dist)

            ## add previously load images in a queue
            total_dist_list = total_dist_list + newly_add_dist_list

            end = time.time()
            lap = end-start
            if i % 50 == 1 :
                print(i, "th : ", lap, " sec")
                kernel_width = np.mean(total_dist_list)
                print(kernel_width)
            if i >= n_image :
                break
            i += 1

        # end for

        kernel_width = np.mean(total_dist_list)
        print(kernel_width)

        return kernel_width, total_dist_list


    # main

    filewritetest()

    kernel_width, total_dist_list = calc_distance(train_loader)

    with open("./dist_list.txt", 'w') as f :
        f.write(str(total_dist_list))


