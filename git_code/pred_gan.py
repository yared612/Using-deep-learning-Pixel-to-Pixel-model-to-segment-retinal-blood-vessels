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

'''load test data'''
test_filename = '/home/kevin/桌面/Retinal/Drive/Data/Training/images_output/'
#test_file_csv = pd.read_csv('/home/yared/文件/ISBI2021/class_2/result/v1/WWW_results.csv')
save_path = '/home/kevin/桌面/Retinal/Drive/Data/pred/v2/'
test_X = glob.glob(test_filename + '*.bmp')
imagearray2 = []
# Change the image path with yours.
for path in np.array(test_X):
    print(path)
    img2 = imread(path)
    img2 = cv2.resize(img2, (512, 512))/255
    # met  = img2.mean()
    # stdt = img2.std()
    # nort = (img2 - met) / stdt
    imagearray2.append(img2)
test_x=np.array(imagearray2)

'''model'''
model_path = '/home/kevin/桌面/Retinal/Drive/Data/weight/v2/generator/generator_393_4.151.h5'
image_shape = (512,512)
model_g = build_generator((image_shape + (3,)))
model_g.load_weights(model_path)

'''predict'''
for tt in range (0,len(test_X)):
    name = test_X[tt].split('/')[-1].split('_')[0]
    im = test_x[tt,:,:,:]
    im1 = np.reshape(im, (1,512,512,3))
    pred = model_g.predict(im1)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    imsave(save_path + name + '.png',pred[0,:,:,:])
