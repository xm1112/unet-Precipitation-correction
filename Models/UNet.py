#!/usr/bin/env python 
# -*- coding:utf-8 -*-


from keras import Model, layers
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPooling2D, concatenate, UpSampling2D, Dropout,Concatenate,Dense
from vortext_pooling import vortex, res_block, res_block_v2
from keras.initializers import random_normal


def UNet(input_height, input_width):
    # assert input_height % 32 == 0
    # assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 1))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block1_conv1')(img_input)  #24*24
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block1_conv2')(x)#
    #x1 = res_block(x1, 64)
    ##x1 = Dropout(rate=0.2)(x1)
    #x1 = vortex(x1, 64)
    x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_1")(x1)  #12*12

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block2_conv1')(x)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block2_conv2')(x)#12*12

    #x2 = vortex(x2, 128)
    x2 = BatchNormalization()(x2)
    print(x2.shape)
    #x2 = Dropout(rate=0.2)(x2)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_2")(x2) #6*6
    print(x.shape)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = random_normal(stddev=0.02),name='block3_conv1')(x)#6*6
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block3_conv2')(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02), name='block3_conv3')(x)#
    x3 = BatchNormalization()(x3)
    print("x3.shape", x3.shape)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="MaxPooling_3")(x3)# 3*3
    print(x.shape)
    print("*************************************")

    o = UpSampling2D((2, 2))(x)#4*4*256
    print(o.shape)
    o = concatenate([x3, o], axis=-1)
    print(o.shape)
    o = Conv2D(256, (3, 3), padding="same", kernel_initializer=random_normal(stddev=0.02))(o)
    o = Conv2D(256, (3, 3), padding="same", kernel_initializer=random_normal(stddev=0.02))(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([x2, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same", kernel_initializer=random_normal(stddev=0.02))(o)
    o = Conv2D(128, (3, 3), padding="same", kernel_initializer=random_normal(stddev=0.02))(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = UpSampling2D((2, 2))(o)#320*320
    o = concatenate([x1, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same",kernel_initializer=random_normal(stddev=0.02))(o)
    o = Conv2D(64, (3, 3), padding="same",kernel_initializer=random_normal(stddev=0.02))(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(1, (1, 1), padding="same",kernel_initializer=random_normal(stddev=0.02))(o)
    o = Dense(1)(o)

    model = Model(inputs=img_input, outputs=o)
    model.summary()
    return model

if __name__ == '__main__':
    m = UNet(24, 24)