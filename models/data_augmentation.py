import math
import numpy as np
import pandas as pd
import multiprocess as mp
import gc
import tensorflow as tf



def get_number_by_percentage(percentage, whole):
    return math.ceil(float(percentage)/100 * float(whole))

"""
input: array [[path_of_file <string>, label <int>]]
output: array of path [path_of_file <string>] & array of label [label <int>]
"""
def selecting_images_preprocessing(images_path_array, limit_image_to_train = "MAX", composition={}):
    # images_path_array = glob(images_path)
    final_image_path = []
    final_label = []
    def processing_image(img_data):
        img_path = img_data[0]
        label = img_data[1]
        # print(img_path, label)
        image = cv2.imread(img_path)
        # print(image)
        mean = np.mean(image)
        std = np.std(image)
        # print(mean, image.mean())
        # print(std, image.std())
        data_row = {
            "image_path": img_path,
            "mean": image.mean(),
            "std": image.std(),
            "class": label
        }
        # print(data_row)
        return data_row
    
        
    print("processed number of data: ", len(images_path_array))
    if limit_image_to_train == "MAX":
        limit_image_to_train = len(images_path_array)
            
    df_analysis = pd.DataFrame(columns=['image_path','mean','std', 'class'])
    
    # multiple processing calculating std
    
    pool = mp.Pool(5)
    data_rows = pool.map(processing_image, images_path_array)
    
    df_analysis = df_analysis.append(data_rows, ignore_index = True)
            
    final_df = df_analysis.sort_values(['std', 'mean'], ascending = [True, False])
    
    if composition == {}:
        final_df = shuffle(final_df)
        final_image_path = final_df['image_path'].head(limit_image_to_train).tolist()
        final_label = final_df['class'].head(limit_image_to_train).tolist()
    else:
        counter_available_no_data = limit_image_to_train
        if composition.get('top') != 0:
            num_rows = get_number_by_percentage(composition.get('top'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows
            
            print(composition.get('top'), num_rows, counter_available_no_data)
            
            # get top data
            final_image_path = final_image_path + final_df['image_path'].head(num_rows).tolist()
            final_label = final_label + final_df['class'].head(num_rows).tolist()
            
        if composition.get('mid') != 0:
            num_rows = get_number_by_percentage(composition.get('mid'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows
            
            print(composition.get('mid'), num_rows, counter_available_no_data)
            
            # top & mid
            n = len(final_df.index)
            mid_n = round(n/2)
            mid_k = round(num_rows/2)

            start = mid_n - mid_k
            end = mid_n + mid_k

            final = final_df.iloc[start:end]
            final_image_path = final_image_path + final['image_path'].head(num_rows).tolist()
            final_label = final_label + final['class'].head(num_rows).tolist()
            
        if composition.get('bottom') != 0:
            num_rows = get_number_by_percentage(composition.get('bottom'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows
            
            print(composition.get('bottom'), num_rows, counter_available_no_data)
            
            # get bottom data
            final_image_path = final_image_path + final_df['image_path'].tail(num_rows).tolist()
            final_label = final_label + final_df['class'].tail(num_rows).tolist()
    
    
    # clear zombies memory
    del [[final_df, df_analysis]]
    gc.collect()
    
    # print(final_image_path, final_label)
    # print(len(final_image_path), len(final_label))
    return final_image_path, final_label

def enhance_image(image, beta=0.1):
    image = tf.cast(image, tf.float64)
    image = ((1 + beta) * image) + (-beta * tf.math.reduce_mean(image))
    # image = ((1 + beta) * image) + (-beta * np.mean(image))
    return image

# ORI_SIZE = (271, 481)
# IMG_H = 128
# IMG_W = 128
# IMG_C = 3  ## Change this to 1 for grayscale.
# winSize = (256, 256)
# stSize = 20


def sliding_crop_and_select_one(img, stepSize=20, windowSize=(256, 256)):
    current_std = 0
    current_image = None
    y_end_crop, x_end_crop = False, False
    for y in range(0, ORI_SIZE[0], stepSize):
        
        y_end_crop = False
        
        for x in range(0, ORI_SIZE[1], stepSize):
            
            x_end_crop = False
            
            crop_y = y
            if (y + windowSize[0]) > ORI_SIZE[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
                y_end_crop = True
            
            crop_x = x
            if (x + windowSize[1]) > ORI_SIZE[1]:
                crop_x = ORI_SIZE[1] - windowSize[1]
                x_end_crop = True
                
            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])                
            std_image = tf.math.reduce_std(tf.cast(image, dtype=tf.float32))
          
            if current_std == 0 or std_image < current_std :
                current_std = std_image
                current_image = image
                
            if x_end_crop:
                break
                
        if x_end_crop and y_end_crop:
            break
            
    return current_image

def sliding_crop(img, stepSize=20, windowSize=(256, 256)):
    current_std = 0
    current_image = []
    y_end_crop, x_end_crop = False, False
    for y in range(0, ORI_SIZE[0], stepSize):
        y_end_crop = False
        for x in range(0, ORI_SIZE[1], stepSize):
            x_end_crop = False
            crop_y = y
            if (y + windowSize[0]) > ORI_SIZE[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
            
            crop_x = x
            if (x + windowSize[1]) > ORI_SIZE[1]:
                crop_x = ORI_SIZE[1] - windowSize[1]
            
            # print(crop_y, crop_x, windowSize)
            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])
            current_image.append(image)
            if x_end_crop:
                break
        if x_end_crop and y_end_crop:
            break
    return current_image

def custom_v3(img):
    img = tf.image.adjust_gamma(img)
    img = tfa.image.median_filter2d(img, 3)
    return img