#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
多输入单输出
"""
from keras import Model, layers
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPooling2D, concatenate, UpSampling2D, Dropout,Concatenate,Dense
#from deform_conv import layers
#from deform_conv.layers import ConvOffset2D
from vortext_pooling import vortex


def UNet_D(nClasses, input_height, input_width):
    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 1))
    dsm_input = Input(shape=(input_height, input_width, 1))
#VGG16特征提取
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)#320*320*128
    x1 = Dropout(rate=0.2)(x1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_1")(x1)

    x2 = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x2)#160*160*128
    #SEnet
    x2 = Dropout(rate=0.2)(x2)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_2")(x2)

    x3 = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x3)#80*80*256
    x3 = Dropout(rate=0.2)(x3)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_3")(x3)

    # x4 = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    # x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x4)
    # x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x4)#40*40*512
    # x4 = SeBlock()(x4)
    # x4 = Dropout(rate=0.2)(x4)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_4")(x4)#20*20*512
    
    #x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)#20*20*1024
    #x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2')(x5)

#*************dsm编码******************************************************************
    x_dsm = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_dsm')(dsm_input)
    x1_dsm = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_dsm')(x_dsm)  # 320*320*128
    x1_dsm = Dropout(rate=0.2)(x1_dsm)
    x_dsm = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_1_dsm")(x1_dsm)

    x_dsm = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_dsm')(x_dsm)
    x2_dsm = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_dsm')(x_dsm)  # 160*160*128
    x2_dsm = Dropout(rate=0.2)(x2_dsm)
    x_dsm = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_2_dsm")(x2_dsm)

    x_dsm = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_dsm')(x_dsm)
    x_dsm = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_dsm')(x_dsm)
    x3_dsm = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_dsm')(x_dsm)  # 80*80*256
    x3_dsm = Dropout(rate=0.2)(x3_dsm)
    x_dsm = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_3_dsm")(x3_dsm)

    # x_dsm = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_dsm')(x_dsm)
    # x_dsm = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2_dsm')(x_dsm)
    # x4_dsm = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3_dsm')(x_dsm)  # 40*40*512
    # x4_dsm = SeBlock()(x4_dsm)
    # x4_dsm = Dropout(rate=0.2)(x4_dsm)
    # x_dsm = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_4_dsm")(x4_dsm)  # 20*20*512
    
    #x5_dsm = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1_dsm')(x_dsm)
    #x5_dsm = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2_dsm')(x5_dsm)
    
#**********************dsm和rgb聚合****************************************************************
    #在编码尾部添加vortex pooling
    #v = vortex(x)

    y3 = Concatenate(axis=-1)([x, x_dsm])
    y3 = Conv2D(256, (1, 1), padding="same")(y3)
    
    #y3 = vortex(y3)
    # 3通道的解码层
    # o = UpSampling2D((2, 2))(y3)#40*40*512
    # o = concatenate([x4, o], axis=-1)
    # o = Conv2D(512, (3, 3), padding="same")(o)
    # o = Conv2D(512, (3, 3), padding="same")(o)
    # o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(y3)#80*80
    o = concatenate([x3, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([x2, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)#320*320
    o = concatenate([x1, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(1, (1, 1), padding="same")(o)
    o = Dense(1)(o)
    # o = BatchNormalization()(o)
    # #o= Dropout(0.5, noise_shape=None, seed=None)
    # o = Activation("relu")(o)#none*320*320*2
    # o = Reshape((-1, nClasses))(o)#  none*1024*2
    # o = Activation("softmax")(o)
    model = Model(inputs=[img_input, dsm_input], outputs=o)
    model.summary()
    return model


if __name__ == '__main__':
    m = UNet(24, 24)