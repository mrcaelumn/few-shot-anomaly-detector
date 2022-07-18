#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tqdm import tqdm
import os
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import tf_clahe

import natsort
from glob import glob
import time
import shutil


# In[ ]:


def resize_image(inputFile, outputFile):
    # print(inputFile)
    img = cv2.imread(inputFile)
    
    img = cv2.resize(img, (481, 271), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite(outputFile, img)
    
def smoothing_func(inputFile, outputFile):
    # print(inputFile)
    img = cv2.imread(inputFile)
    
    # img = cv2.GaussianBlur(img,(7,7),0)
    img = cv2.blur(img, (7,7))
    
    cv2.imwrite(outputFile, img)
    
def convert_file_clahe(inputFile, outputFile):
    bgr = cv2.imread(inputFile)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    # img = clahe.apply(lab)
    eq_channels = []
    for ch in lab_planes:
        eq_channels.append(clahe.apply(ch))

    eq_image = cv2.merge(eq_channels)

    bgr = cv2.cvtColor(eq_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(outputFile, bgr)
    
    
def convert_custom_method(inputFile, outputFile):
    img = tf.io.read_file(inputFile)
    img = tf.io.decode_png(img, channels=3)
    # print(tf.rank(img))
    # img = tf.cast(img, tf.float32)

    # img = tf.image.adjust_gamma(img)
    # img = tfa.image.equalize(img)
    # img = tfa.image.gaussian_filter2d(img)
    img = tfa.image.mean_filter2d(img, filter_shape=(5, 5))
    # img = tfa.image.median_filter2d(img, filter_shape=(5, 5))
    
    
    tf.keras.utils.save_img(outputFile, img)

def convert_file_clahe_v2(inputFile, outputFile):
    img = tf.io.read_file(inputFile)
    img = tf.io.decode_png(img, channels=3)
    
    img = tf_clahe.clahe(img, tile_grid_size=(8, 8), clip_limit=3)
    
    tf.keras.utils.save_img(outputFile, img)


# In[ ]:


# path = 'data/mura_april/train_data/normal'
# files = os.listdir(path)

# for index, file in enumerate(files):
#     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.png'])))
# print("Done.")

def remove_hidden_file(list_arr):
    if '.DS_Store' in list_arr:
        list_arr.remove('.DS_Store')

    if '.ipynb_checkpoints' in list_arr:
        list_arr.remove('.ipynb_checkpoints')
    return list_arr


# In[ ]:


root_target_dir = "data/mura_april_linear_blurred_v3"
root_folder = "data/mura_april_linear"
list_root_dir = os.listdir(root_folder)
# print(list_root_dir)
list_root_dir = remove_hidden_file(list_root_dir)
# print(list_root_dir)

for mode in list_root_dir:
    for class_name in ["normal", "defect"]:
    # for class_name in ["mura", "defect"]:
        Input_dir = f'{root_folder}/{mode}/{class_name}/'
        Out_dir = f'{root_target_dir}/{mode}/{class_name}/'
        # create dir if it isnt exist
        if not os.path.exists(Out_dir):
            os.makedirs(Out_dir)

        for i in tqdm(os.listdir(Input_dir), desc=f'Converting dataset {mode} with class {class_name} images {root_target_dir}'):
            if i != ".DS_Store" and i != ".ipynb_checkpoints":

                inputFile = Input_dir+i
                outputFile = Out_dir+i
                # print(inputFile)
                # resize_image(inputFile, outputFile)
                smoothing_func(inputFile, outputFile)
                # convert_file_clahe(inputFile, outputFile)
                # convert_file_clahe_v2(inputFile, outputFile)
                # convert_custom_method(inputFile, outputFile)
                


# In[ ]:


print("test")


# In[ ]:




