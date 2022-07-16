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
    

def convert_sobel(inputFile, outputFile, x=1, y=1, ks=3, depth=cv2.CV_32F):
    img = cv2.imread(inputFile)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.Sobel(img, cv2.CV_16U, x, y,ksize=ks)
    img = cv2.Sobel(img, depth, x, y,ksize=ks)
    # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(outputFile, img)
    
def convert_scharr(inputFile, outputFile, x=1, y=1, depth=cv2.CV_32F):
    img = cv2.imread(inputFile)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.Scharr(img, depth, x, y)
    # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
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


# ## convert colour of images
# # for target_folder in [
# #     "mura_sobelx_16","mura_sobely_16", "mura_sobelxy_16", 
# #     "mura_sobelx_16_v2","mura_sobely_16_v2", "mura_sobelxy_16_v2"]:

# root_target_dir = "target_data_nearest"
# list_target_dir = natsort.natsorted(os.listdir(root_target_dir))

# list_target_dir = remove_hidden_file(list_target_dir)
# # print(list_target_dir)

# # ['mura', 'sobel', 'xy', 'ori', 'v2']

# for target_folder in list_target_dir:  
#     target_data = target_folder.split("_")
#     print(target_data)
#     type_operator, type_pre, depth, kersize = target_data[1], target_data[2], target_data[3], target_data[4]
    
#     x, y, ks, dp = 1, 1, 1, -1
    
#     if type_pre == "x":
#         y=0
#     elif type_pre == "y":
#         x=0
    
#     if depth == "8":
#         dp=cv2.CV_8U
#     elif depth == "16":
#         dp=cv2.CV_16U
#     elif depth == "32":
#         dp=cv2.CV_32F
#     elif depth == "64":
#         dp=cv2.CV_64F
    
#     if kersize == "v3":
#         ks=3
#     elif kersize == "v5":
#         ks=5
    
        
#     print(f"{type_operator=}, {type_pre=}, {depth=}, {kersize=}")
#     print(f"{x=}, {y=}, {ks=}, {dp=}")
    
#     # time.sleep(3)
#     # for mode in ["test_data", "test_data_v2","train_data", "eval_data"]:
#     root_folder = "resize_data/mura_april_nearest"
#     list_root_dir = os.listdir(root_folder)
#     # print(list_root_dir)
#     list_root_dir = remove_hidden_file(list_root_dir)
#     # print(list_root_dir)
        
#     for mode in list_root_dir:
#         for class_name in ["normal", "defect"]:
#         # for class_name in ["mura", "normal", "smura"]:
#             Input_dir = f'{root_folder}/{mode}/{class_name}/'
#             Out_dir = f'{root_target_dir}/{target_folder}/{mode}/{class_name}/'
#             # create dir if it isnt exist
#             if not os.path.exists(Out_dir):
#                 os.makedirs(Out_dir)
                
#             for i in tqdm(os.listdir(Input_dir), desc=f'Converting dataset {mode} with class {class_name} images {target_folder}'):
                
#                 if i != ".DS_Store" and i != ".ipynb_checkpoints":
                    
#                     inputFile = Input_dir+i
#                     outputFile = Out_dir+i

#                     if type_operator == "sobel":
#                         # print("sobel function executed.")
#                         convert_sobel(inputFile, outputFile, x=x, y=y, ks=ks, depth=dp)
#                     elif type_operator == "scharr":
#                         # print("scharr function executed.")
#                         convert_scharr(inputFile, outputFile, x=x, y=y, depth=dp)

#             # print("done.", class_name, mode)
#         #     break


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




