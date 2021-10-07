from keras.losses import binary_crossentropy,mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from skimage.io import imread,imsave,imshow
from keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import backend as K
from modelp2p import *
from tqdm import tqdm
import pandas as pd
import keras as ks
import numpy as np
import glob
import pdb
import cv2
import os

'''path & read'''
# data_gt  = pd.read_csv('/home/yared/文件/ISBI2021/Training_Set/RFMiD_Training_Labels.csv')
filename = './Training/images_output/'
filename_gt = './Training/1st_manual_output/'
# modelg_path = '/home/kevin/桌面/Retinal/Drive/Data/weight/v1/generator/s1/generator_90_4.942.h5'
# modeld_path = '/home/kevin/桌面/Retinal/Drive/Data/weight/v1/discriminator/s1/discriminator_90_4.985.h5'
# test     = glob.glob('/home/yared/文件/ISBI2021/test/*.png')
version  = 'v3'

'''Build dir'''
if not os.path.isdir('./weight'):
    os.mkdir('./weight')
if not os.path.isdir('./weight/' + version):
    os.mkdir('./weight/' + version)
if not os.path.isdir('./weight/' + version + '/' + 'generator'):
    os.mkdir('./weight/' + version + '/' + 'generator')
if not os.path.isdir('./weight/' + version + '/' + 'discriminator'):
    os.mkdir('./weight/' + version + '/' + 'discriminator')
if not os.path.isdir('./record'):
    os.mkdir('./record')
if not os.path.isdir('./record/' + version):
    os.mkdir('./record/' + version)

'''Make Training & validation data'''
im_files = glob.glob(filename + '*.bmp')
im_files.sort()
train_y = []
for  x_n in im_files:
    name = x_n.split('/')[-1].split('_')[0]
    train_y.append(filename_gt + name + '_manual1.bmp')
# im_files, valid_im, train_y, valid_y = train_test_split(X,y,test_size=0.05,random_state=0,shuffle=True)
'''Hyperparameter'''
image_shape = (512,512)
batch_size = 1
epochs = 500
learning_rate = 0.0001
weight_decay = 1e-9

'''Make Training & validation data'''
           
def read_data(im_files,label_files):
    x,y = [],[]
    for im_file,label_file in zip(im_files,label_files):
        img = imread(im_file).astype(np.float)
        im = cv2.resize(img, image_shape)/255
        img_b = imread(label_file).astype(np.float)
        im_b = cv2.resize(img_b, image_shape )/255
        m1,n1 = im_b.shape
        im_bb = np.zeros((m1,n1,1))
        im_bb[:,:,0] = im_b
        # label = np.reshape(label,label.shape + (1,))
        # label = np.concatenate([label,label,label],axis=-1)
        
        #im = (im-np.std(im))/np.mean(im)
        x.append(im)
        y.append(im_bb)
    y  = np.stack(y, axis = 0)
    x  = np.stack(x, axis = 0)

    return x , y

def data_gen_fn(im_files,label_files,batch_size):
    Im,label = shuffle(im_files,train_y)
    i = 0
    while True:
        start = i * batch_size
        if (start + batch_size) > len(Im):
            end   = len(Im)
            start2= 0
            end2  = start + batch_size - len(Im)
            Ims   = Im[start:end]   + Im[start2:end2]
            labels= label[start:end]+ label[start2:end2]
            # debug_(Ims,labels)
            yield read_data(Ims,labels)
        else :
            end   = start + batch_size
            # debug_(Im[start:end],label[start:end])
            yield read_data(Im[start:end],label[start:end])
        
        i = i + 1
        if (i * batch_size) >= len(Im):
            Im,label = shuffle(im_files,train_y)
            i = 0

'''Build model'''
data_gen = data_gen_fn(im_files,train_y,batch_size)
steps_per_epochs = int(np.ceil(len(im_files) / batch_size))
model_d = build_discriminator(image_shape + (3,),image_shape + (1,))
model_g = build_generator((image_shape + (3,)))
# model_g.load_weights(modelg_path)
# model_d.load_weights(modeld_path)
d_end = model_d.output.shape.as_list()
combine = combined(model_g,model_d,(image_shape + (3,)),image_shape + (1,))
model_d.summary()
optimizer = Adam(lr=learning_rate, decay=weight_decay)
model_d.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
combine.compile(loss=['mse', 'binary_crossentropy'],loss_weights=[1, 100],optimizer=optimizer)

'''Training'''
find_best_name=['Epoch','D_loss','D_acc','G_loss']
find_best = []
best_g_loss = 0
patch = int(image_shape[0] / 2**4)
disc_patch = (patch, patch, 1)
for epoch in range(epochs):
    valid = np.ones((batch_size,) + disc_patch)
    fake  =np.zeros((batch_size,) + disc_patch)
    for steps_per_epoch in tqdm(range(steps_per_epochs)):
        '''
        Train Discriminator
        '''
        imgs_A,imgs_B = next(data_gen)
        fake_A = model_g.predict(imgs_A)
        d_loss_real = model_d.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = model_d.train_on_batch([imgs_A, fake_A], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        '''
        Train Generator
        '''
        g_loss = combine.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])
        
        if epoch == 0:
            best_g_loss = g_loss[0]

    print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] " % (epoch, epochs,
                                                        d_loss[0], 100*d_loss[1],
                                                        g_loss[0]))
    find_best.append([epoch,d_loss[0],100*d_loss[1],g_loss[0]])
    
    if  g_loss[0] < best_g_loss:
        best_g_loss = g_loss[0]
        print('Model saved!')
        dis_name = 'discriminator_%d_%.3f.h5' % (epoch,d_loss[0])
        gen_name = 'generator_%d_%.3f.h5' % (epoch,g_loss[0])
        model_d.save('./weight/' + version + '/' + 'discriminator/' + dis_name)
        model_g.save('./weight/' + version + '/' + 'generator/' + gen_name)
    
pd.DataFrame(columns=find_best_name,data=find_best).to_csv('./record/' + version + '/gan_record.csv',index=False,encoding='gbk')



