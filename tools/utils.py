from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import imageio
import numpy as np
from time import gmtime, strftime
import cv2

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)
from PIL import Image
import matplotlib.pyplot as plt


def save_images(images, size, image_path):
    images=inverse_transform(images)
   # images=images[1:]
    x=images.shape
   # images=images[0,:,:,:]
    #data = np.zeros((32, 128, 128, 3),dtype=np.uint8)
    #data[0, :, :, :] = np.asarray(images)
    #im = Image.fromarray(np.dot(data[0], [0.299, 0.587, 0.114]))
   # im=Image.fromarray(np.asarray(images), 'RGB')
    #im.show()
    #plt.imshow(images)
   # data_new = np.rollaxis(images, 3, 1).reshape(m, -1, n)
    #data = np.ones((1, 8, 8, 3))
    #for i in range(8):
        #data[0, i, i, 1] = 0.0

    #print("size: %s, type: %s" % (data.shape, data.dtype))
    # size: (1, 16, 16, 3), type: float64

    data_img = (images.squeeze() * 255).astype(np.uint8)

    print("size: %s, type: %s" % (data_img.shape, data_img.dtype))
    # size: (16, 16, 3), type: uint8

    img = Image.fromarray(data_img, mode='RGB')
    #img.show()
    return img.save(image_path)
    return imageio.imwrite(img,size, image_path)

def save_images2(images, size, image_path):
    return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def save_source(images, size, path):
    img = merge(images, size)
    mean = np.array([104., 117., 124.])
    img = np.uint8(img + mean)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.