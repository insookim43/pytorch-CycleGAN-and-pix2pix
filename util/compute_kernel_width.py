import numpy
import hsic
import cv2

import numpy as np

import glob
import time


filename = r"C:\Users\Kim\Desktop\dataset\CIFAR10-ImageNet_codetest_32x32\trainA\*\*.png"
filenames = glob.glob(filename)
# filenames.sort()


# number of images
n_image = 20000

# helper functions
def compute_distances(order, im):
    load_image_before = order
    newly_add_dist_list = []
    for i in range(load_image_before):
        previous_image = cv2.imread(filenames[i])
        previous_image = np.reshape(-1,1)
        temp_l2_dist = np.linalg.norm(im - previous_image)
        newly_add_dist_list.append(temp_l2_dist)
    return newly_add_dist_list

def load_and_preprocess():
    pass


def calc_distance(filenames):
    pass

total_dist_list= []

print(glob.glob(filename)[0])
# compute distance
for i,img in enumerate(filenames):
    start = time.time()
    ## load new image
    im = cv2.imread(img)
#    print("im.shape", im.shape)
    im = im.reshape([1, -1])

    ## compute 2d distances between previous load images
    computed_distances = compute_distances(i, im)
    ## add previously load images in a queue
    total_dist_list = total_dist_list + computed_distances

    end = time.time()
    lap = end-start
    if i % 50 == 1 :
        print(i, "th : ", lap, " sec")
        kernel_width = np.mean(total_dist_list)
        print(kernel_width)
    if i >= n_image :
        break

# end for

kernel_width = np.mean(total_dist_list)
print(kernel_width)


