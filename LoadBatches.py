
"""
2021/4/21

"""

import numpy as np
import glob
import itertools
import matplotlib.pyplot as plt
import random
from PIL import Image
import sys
np.set_printoptions(threshold=np.inf)

def getImageArr(im):
    # img = im.astype(np.float32)
    # a = np.max(img)
    # b = np.min(img)
    # img = (im-b)/(a-b)
    # im2 = im/500.0
    return im


def imageSegmentationGenerator(images_path, segs_path, batch_size,
                                input_height, input_width):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.tif"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.tif"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        try:
            X = []
            Y = []
            for _ in range(batch_size):
                im, seg = zipped.__next__()

                im = Image.open(im)
                im = im.resize((96,96))
                im = np.array(im)
                #print(im.shape)

                seg = Image.open(seg)
                seg = seg.resize((96,96))
                seg = np.array(seg)

                assert im.shape[:2] == seg.shape[:2]
                assert im.shape[0] >= input_height and im.shape[1] >= input_width

                xx = random.randint(0, im.shape[0] - input_height)
                yy = random.randint(0, im.shape[1] - input_width)

                im = im[xx:xx + input_height, yy:yy + input_width]
                seg = seg[xx:xx + input_height, yy:yy + input_width]

                X.append(getImageArr(im))
                Y.append(getImageArr(seg))

            X = np.expand_dims(X, axis=3)
            Y = np.expand_dims(Y, axis=3)
            print("x：", X.shape)
            yield np.array(X), np.array(Y)
        except StopIteration:
            sys.exit()
            #break


if __name__ == '__main__':
    G = imageSegmentationGenerator("E:/14DL-unet-Regression/run/datasets/led2rawmean/train_data/led2raw/",
                                   "E:/14DL-unet-Regression/run/datasets/led2rawmean/train_data/obsn/", batch_size=12,  input_height=24, input_width=24)

    x, y = G.__next__()


