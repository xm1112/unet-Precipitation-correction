
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import LoadBatches
from Models import FCN8, FCN32, SegNet, UNet
from keras import optimizers
import math
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.optimizers import Adam
import keras
from keras.callbacks import EarlyStopping
tf.set_random_seed(1234)
#############################################################################

train_images_path = "E:/14DL-unet-Regression/run/datasets/led2rawmean_2/train_data/raw/"
train_segs_path = "E:/14DL-unet-Regression/run/datasets/led2rawmean_2/train_data/obsn/"
train_batch_size = 12

epochs = 150

input_height = 24
input_width = 24

val_images_path = "E:/14DL-unet-Regression/run/datasets/led2rawmean_2/val_data/raw/"
val_segs_path = "E:/14DL-unet-Regression/run/datasets/led2rawmean_2/val_data/obsn/"
val_batch_size = 12

weight_path = "E:/14DL-unet-Regression/run/log/led2rawmean_2/b12-e"+str(epochs)+"-mask-w"
if os.path.exists(weight_path):
    pass
else:
    os.mkdir(weight_path)


key = "unet"
method = {
    "fcn32": FCN32.FCN32,
    "fcn8": FCN8.FCN8,
    'segnet': SegNet.SegNet,
    'unet': UNet.UNet}

m = method[key](input_height=input_height, input_width=input_width)
def cc(yTrue, yPred):
    print(yTrue.shape)
    print(yPred.shape)
    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print(W)
    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)
    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)


    d1 = tf.zeros_like(yPred)
    yPred = tf.where(tf.less(yPred, 0), d1, yPred)
    #cc
    y_true_mean = (1/225)*tf.reduce_sum(W*yTrue)
    print(y_true_mean)
    y_pred_mean = (1/225)*tf.reduce_sum(W*yPred)

    s = tf.reduce_sum(W*(yTrue - y_true_mean) * (yPred - y_pred_mean))
    t = K.sqrt(tf.reduce_sum(W*K.square(yTrue - y_true_mean)))
    p = K.sqrt(tf.reduce_sum(W*K.square(yPred - y_pred_mean)))
    print(p)
    cc = s / (t * p)
    return cc

def weighted_mse(yTrue,yPred):
    # 真实的标签
    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print(W)
    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)
    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)
    print(W)  #(24,24)
    #return (1/225)*tf.reduce_sum(W*K.square(yTrue-yPred))
    return K.mean(W * K.square(yTrue - yPred))

def weighted_mse2(yTrue,yPred):
    print(yTrue.shape)
    print(yPred.shape)
    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print(W)
    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)
    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)
    print(W)
    d1 = tf.zeros_like(yPred)
    print(d1)
    yPred = tf.where(tf.less(yPred, 0), d1, yPred)
    print(yPred) #Tensor("loss/dense_1_loss/Select:0", shape=(?, 24, 24, 1), dtype=float32)

    return (1/225)*tf.reduce_sum(W*K.square(yTrue-yPred))


def weighted_mse3(yTrue,yPred):

    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print(W)
    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)
    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)
    print(W)

    # num_list = [[500] * 24 for j in range(24)]
    # num_list = np.array(num_list)
    # a = tf.convert_to_tensor(num_list, dtype=tf.float32)
    # a = tf.expand_dims(a, 0)
    # a = tf.expand_dims(a, 3)
    # print(a)

    #x = tf.placeholder(tf.float32, shape=[None, 24, 24,1])
    d1 = tf.ones_like(yPred)
    d2 = tf.multiply(d1, 500, name=None)
    # d2 = tf.expand_dims(d2, 3)
    print("d2", d2)
    W1 = tf.ones_like(yPred)
    W1 = tf.multiply(W, W1, name=None)
    print("W1", W1)
    yPred = tf.where(tf.equal(W1, 0), d2, yPred)
    print(yPred) #
    return (1/225)*tf.reduce_sum(K.square(yTrue-yPred))


def weighted_msemae(yTrue,yPred):
    # 真实的标签
    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print(W)
    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)
    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)
    print(W)  #(24,24)
    d1 = tf.zeros_like(yPred)
    yPred = tf.where(tf.less(yPred, 0), d1, yPred)
    return (1/225)*tf.reduce_sum(W*K.square(yTrue-yPred))+(1/225)*K.mean(K.abs(yPred - yTrue), axis=-1)


def mask_weighted_mse(yTrue,yPred):
    print(yTrue.shape)
    print(yPred.shape)
    img2 = Image.open("E:/14DL-unet-Regression/run/datasets/led1rawmean/test_data/obsn/920.tif")
    img2 = np.array(img2)
    img2 = img2.reshape(-1)
    W = []
    for i in range(24 * 24):
        if img2[i] == 500:
            W.append(0)
        else:
            W.append(1)
    W = np.array(W).astype(float)
    W = W.reshape(24, 24)
    print("-----------------")
    print(W)
    print(W.shape)  #(24, 24)
    print("-----------------")

    W = tf.constant(W)
    W = tf.cast(W, dtype=tf.float32)

    W = tf.expand_dims(W, 0)
    W = tf.expand_dims(W, 3)
    print(W)  #(1,24,24,1)
    d1 = tf.zeros_like(yPred)
    print(d1) #(?, 24, 24, 1)

    yPred = tf.where(tf.less(yPred, 0), d1, yPred)
    # print(yPred) #Tensor("loss/dense_1_loss/Select:0", shape=(?, 24, 24, 1), dtype=float32)
    w1 = tf.multiply(tf.ones_like(yTrue), 0.1)# tf.cast(tf.constant(1), tf.float32)
    w2 = tf.multiply(tf.ones_like(yTrue), 1)#tf.cast(tf.constant(2), tf.float32)
    w3 = tf.multiply(tf.ones_like(yTrue), 10)#tf.cast(tf.constant(5), tf.float32)
    w4 = tf.multiply(tf.ones_like(yTrue), 25)# tf.cast(tf.constant(10), tf.float32)
    w5 = tf.multiply(tf.ones_like(yTrue), 50)#tf.cast(tf.constant(30), tf.float32)
    print(w5)
    print(w5.shape)
    t1 = tf.cast(tf.constant(50), dtype=tf.float32)
    t3 = tf.cast(tf.constant(10), dtype=tf.float32)
    t4 = tf.cast(tf.constant(1), dtype=tf.float32)

    f4 = tf.where(tf.less(yTrue, t1),  w4,  w5)
    f3 = tf.where(tf.less(yTrue, t2),  w3,  f4)
    f2 = tf.where(tf.less(yTrue, t3),  w2,  f3)
    we = tf.where(tf.less(yTrue, t4),  w1,  f2)

    we_mse = (1/225)*tf.reduce_sum(we*K.square(W*yTrue-W*yPred))
    return we_mse

#loss = keras.losses.huber_loss(y_true, y_pred, delta=1.0)
m.compile(
    loss="mse",
    optimizer=Adam(lr=0.001),
    metrics=['acc', 'mse'])


G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                           train_segs_path, train_batch_size, input_height=input_height, input_width=input_width)
G_test = LoadBatches.imageSegmentationGenerator(val_images_path,
                                                val_segs_path, val_batch_size,  input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath=weight_path+"/%s_model.h5" %
    key,
    monitor='val_loss',
    mode='auto',
    save_best_only='True')
tensorboard = TensorBoard(log_dir=weight_path+'/event/log_%s_model' % key)


history = m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(828./train_batch_size),
                epochs=epochs, callbacks=[checkpoint, tensorboard],
                verbose=2,
                validation_data=G_test,
                validation_steps=1,
                class_weight=None,
                shuffle=True)

accy = history.history['acc']
lossy = history.history['loss']
np_accy = np.array(accy).reshape((1, len(accy)))
np_lossy = np.array(lossy).reshape((1, len(lossy)))
np_out = np.concatenate([np_accy, np_lossy], axis=0)
np.savetxt(weight_path+'/log_%s_model.txt'% key, np_out)
print("save")



