#!/usr/bin/env python
# -*- coding:utf-8 -*-
import LoadBatches
from keras.models import load_model
from Models import FCN32, FCN8, SegNet, UNet
import glob
import itertools
import cv2
import numpy as np
import random
import os
from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plt

#忽略警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

key = "unet"

method = {
    "fcn32": FCN32.FCN32,
    "fcn8": FCN8.FCN8,
    "segnet": SegNet.SegNet,
    'unet': UNet.UNet}

images_path = "E:/14DL-unet-Regression/run/datasets/led7rawmean_2/test_data/raw/"
segs_path = "E:/14DL-unet-Regression/run/datasets/led7rawmean_2/test_data/obsn/"

IMAGE_DIR = images_path
count = os.listdir(IMAGE_DIR)
print( len(count))

SEG_DIR = segs_path

input_height = 24
input_width = 24


def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    # short_edge = min(shape[1:3])
    # xx = int((shape[2] - short_edge) / 2)
    # yy = int((shape[1] - short_edge) / 2)
    return xx, yy

images = sorted(glob.glob(images_path +"*.jpg") +glob.glob(images_path +"*.tif") +glob.glob( images_path +"*.png"))
segmentations = sorted(glob.glob(segs_path + "*.jpg") +glob.glob(segs_path + "*.tif") + glob.glob(segs_path + "*.png"))
print(images)
print(segmentations)
m = method[key](24, 24)
m.load_weights("E:/14DL-unet-Regression/run/log/led7rawmean_2/b12-e150-final/%s_model.h5" % key)

for i, (imgName, segName) in enumerate(zip(images, segmentations)):
    print('success')
    print("%d/%d %s" % (i + 1, len(images), imgName))
    f = imgName.split("/")[-1]
    s = segName.split("/")[-1]
    f1= f.split("\\")[-1]
    print(f)
    print(f1)
    print(f[4:10])
    im = Image.open(imgName)
    im = np.array(im)
    print(im)
    print(im.shape)

    xx, yy = getcenteroffset(im.shape, input_height, input_width)
    print("xx: ", xx)
    print("yy: ", yy)

    seg = Image.open(segName)
    seg = np.array(seg)

    im = np.expand_dims(LoadBatches.getImageArr(im), 0)
    im = np.expand_dims(im, 3)
    pr = m.predict(im)[0]

    print(pr.shape)
    pr =pr.reshape(input_height, input_width)
    print("**************pr**************")
    print(pr)
    print(pr.shape)

    pr2 = Image.fromarray(pr)
    pr2.save("E:/14DL-unet-Regression/run/datasets/led7rawmean_2/test_data/pre-result/b12-e150-final/"+f1)





