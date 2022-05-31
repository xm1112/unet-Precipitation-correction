#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@author: LiShiHang
@software: PyCharm
@file: UNet.py
@time: 2018/12/27 16:54
@desc:
"""
import keras
from keras import Model, layers
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPooling2D, concatenate, UpSampling2D, Dropout,Concatenate,Dense


def UNet(input_height, input_width):
    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 1))
#VGG16特征提取
    x = Conv2D(64, (3, 3), activation='relu', padding='same',   name='block1_conv1')(img_input)  #
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same',  name='block1_conv2')(x)#
    #x1 = Dropout(rate=0.5)(x1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_1")(x1)  #
    print(x.shape)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same',  name='block2_conv2')(x)
    print(x2.shape)
    #x2 = Dropout(rate=0.5)(x2)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_2")(x2) #
    print(x.shape)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2')(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3')(x)#4*5*256
    #x3 = Dropout(rate=0.5)(x3)
    print("x3.shape", x3.shape)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_3")(x3)#2, 2, 256
    print(x.shape)

    # x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)#40*40*512
    # #x4 = Dropout(rate=0.2)(x4)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_4")(x4)#20*20*512
    print("**************************************")
    o = UpSampling2D((2, 2))(x)#4*4*256
    print(o.shape)
    o = concatenate([x3, o], axis=-1)
    print(o.shape)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    #o = Dropout(rate=0.5)(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([x2, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    #o = Dropout(rate=0.5)(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)#320*320
    o = concatenate([x1, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    #o = Dropout(rate=0.5)(o)
    o = BatchNormalization()(o)

    o = Conv2D(1, (1, 1), padding="same")(o)
    o = Dense(1)(o)

    model = Model(inputs=img_input, outputs=o)
    model.summary()
    return model


if __name__ == '__main__':
    m = UNet(24, 24)