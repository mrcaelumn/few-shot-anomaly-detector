#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tqdm import tqdm
import os
import cv2

import natsort
from glob import glob
import time
import shutil


# In[ ]:


def resize_image(inputFile, outputFile):
    # print(inputFile)
    img = cv2.imread(inputFile)
    
    img = cv2.resize(img, (481, 271), interpolation = cv2.INTER_LANCZOS4)
    
    cv2.imwrite(outputFile, img)


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


root_target_dir = "target_data/mura_data_c"
root_folder = "data/mura_type_c"
list_root_dir = os.listdir(root_folder)
# print(list_root_dir)
list_root_dir = remove_hidden_file(list_root_dir)
# print(list_root_dir)

for mode in list_root_dir:
    # for class_name in ["normal", "defect"]:
    for class_name in ["mura", "defect"]:
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
                resize_image(inputFile, outputFile)


# In[ ]:


print("test")


# In[ ]:




