from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np

def conv2d(layer_input, filters, f_size=4, bn=True):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d
def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = Concatenate()([u, skip_input])
    return u

def d_layer(layer_input, filters, f_size=4, bn=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d

def build_generator(img_shape = (512,512,3),gf=64):
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

    return Model(d0, output_img)

def build_discriminator(imgA_shape,imgB_shape,df = 64):

    img_A = Input(shape=imgA_shape)
    img_B = Input(shape=imgB_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

def combined(generator, discriminator,img_shape_A,img_shape_B,optimizer= Adam(0.0002, 0.5)):
    img_A = Input(shape=img_shape_A)
    img_B = Input(shape=img_shape_B)
    fake_A = generator(img_A)
    discriminator.trainable = False
    valid = discriminator([img_A, fake_A])
    return Model(inputs=[img_A, img_B], outputs=[valid, fake_A])

# g = build_generator()
# d = build_discriminator(imgA_shape=(256,256,3),imgB_shape=(256,256,3),df = 64)
# d.summary()
# ss = combined(g,d,img_shape_A = (256,256,3),img_shape_B = (256,256,3),optimizer= Adam(0.0002, 0.5))
# ss.summary()
