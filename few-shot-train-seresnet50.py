#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import tensorflow_addons as tfa
import itertools

import os
from tqdm import tqdm
import numpy as np
import random
import gc
import multiprocess as mp
import pandas as pd 

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime


# In[ ]:


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, MaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, Concatenate, Input, Dense, Reshape, Multiply, add, Flatten, ZeroPadding2D
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K


# In[ ]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

ORI_SIZE = (271, 481)
IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.
winSize = (256, 256)
stSize = 20
# Weight initializers for the Generator network
# WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
AUTOTUNE = tf.data.AUTOTUNE

TRAIN = True

LIMIT_EVAL_IMAGES = 100
LIMIT_TEST_IMAGES = "MAX"
LIMIT_TRAIN_IMAGES = 100

# range between 0-1
anomaly_weight = 0.7
learning_rate = 0.002
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
inner_iters = 4

train_shots = 100
shots = 20
classes = 1
n_shots = shots
if shots > 20 :
    n_shots = "few"
    
dataset_name = "mura"
eval_dataset_name = "mura"
test_dataset_name = "mura"

mode_colour = str(IMG_H) + "_rgb"
if IMG_C == 1:
    mode_colour = str(IMG_H) + "_gray"

model_type = "seresnet50"
name_model = f"{mode_colour}_{dataset_name}_{model_type}_{n_shots}_shots_mura_detection_{str(meta_iters)}"
g_model_path = f"saved_model/{name_model}_g_model.h5"
d_model_path = f"saved_model/{name_model}_d_model.h5"


train_data_path = f"data/{dataset_name}/train_data"
eval_data_path = f"data/{eval_dataset_name}/eval_data"
test_data_path = f"data/{test_dataset_name}/test_data"


# In[ ]:


# class for SSIM loss function
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self,
         reduction=tf.keras.losses.Reduction.AUTO,
         name='SSIMLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, ori, recon):
        recon = tf.convert_to_tensor(recon)
        ori = tf.cast(ori, recon.dtype)

        # Loss 3: SSIM Loss
#         loss_ssim =  tf.reduce_mean(1 - tf.image.ssim(ori, recon, max_val=1.0)[0]) 
        loss_ssim = tf.reduce_mean(1 - tf.image.ssim(ori, recon, max_val=IMG_W, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))
        return loss_ssim
    

class MultiFeatureLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='FeatureLoss'):
        super().__init__(reduction=reduction, name=name)
        self.mse_func = tf.keras.losses.MeanSquaredError() 

    
    def call(self, real, fake, weight=1):
        result = 0.0
        for r, f in zip(real, fake):
            result = result + (weight * self.mse_func(r, f))
        
        return result
    
    
# class for Adversarial loss function
class AdversarialLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='AdversarialLoss'):
        super().__init__(reduction=reduction, name=name)

    
    def call(self, logits_in, labels_in):
        labels_in = tf.convert_to_tensor(labels_in)
        logits_in = tf.cast(logits_in, labels_in.dtype)
        # Loss 4: FEATURE Loss
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


# In[ ]:


''' calculate the auc value for lables and scores'''
def roc(labels, scores, name_model):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    # print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshod that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    # plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_threshold


# In[ ]:


# delcare all loss function that we will use
# L1 Loss
mae = tf.keras.losses.MeanAbsoluteError()
# L2 Loss
mse = tf.keras.losses.MeanSquaredError() 

multimse = MultiFeatureLoss()
# SSIM loss
ssim = SSIMLoss()


# In[ ]:


class GCAdam(tf.keras.optimizers.Adam):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


# In[ ]:


def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
    filename = f"samples/generated_plot_epoch-{epoch}.png"
    plt.savefig(filename)
    plt.close()


# In[ ]:


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'_cm.png')
    plt.show()
    plt.clf()


# In[ ]:


def plot_epoch_result(iters, loss, name, model_name, colour):
    plt.plot(iters, loss, colour, label=name)
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name+ '_'+name+'_iters_result.png')
    plt.show()
    plt.clf()

def plot_anomaly_score(anomaly_scores, name, model_name):
    for key, val in anomaly_scores.items():
        sns.distplot(val,  kde=False, label=key)
    
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(model_name+ '_'+name+'_anomay_scores_dist.png')
    plt.show()
    plt.clf()
    
def enhance_image(image, beta=0.1):
    image = tf.cast(image, tf.float64)
    image = ((1 + beta) * image) + (-beta * tf.math.reduce_mean(image))
    return image

def selecting_images_preprocessing(images_path_array, limit_image_to_process="MAX", limit_image_to_train = "MAX", middle_rows=False):
    # images_path_array = glob(images_path)
    final_image_path = []
    def processing_image(img_path):
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
            # "class": 0
        }
        return data_row
    original_number_number_image = len(images_path_array)
    
    print("original number of data: ", original_number_number_image)
    
    if limit_image_to_train == "MAX":
        limit_image_to_train = original_number_number_image
    
    if limit_image_to_process == "MAX":
        print("You choose to use all of data. please wait it will take a moment.")
    elif len(images_path_array) < limit_image_to_process:
        print("The amount of dataset smaller than limit so we will use all images in dataset.")
    else:
        images_path_array = sample(images_path_array,limit_image_to_process)
    print("processed number of data: ", len(images_path_array))
    
    df_analysis = pd.DataFrame(columns=['image_path','mean','std'])
    counter = 0
    
    start_time = datetime.now()
    # multiple processing
    pool = mp.Pool(5)
    data_rows = pool.map(processing_image, images_path_array)
    # do your work here
    
    end_time = datetime.now()
    print(f'(selecting_images_preprocessing) Duration of counting std and mean of images: {end_time - start_time}')
    # print(data_rows)
    
    df_analysis = df_analysis.append(data_rows, ignore_index = True)
    # counter += 1
    # if counter % 100 == 0:
    #     print("processed image: ", counter)
            
    final_df = df_analysis.sort_values(['std', 'mean'], ascending = [True, False])
    
    
    if middle_rows:
        print("get data from middle row")
        n = len(final_df.index)
        mid_n = round(n/2)
        mid_k = round(limit_image_to_train/2)


        start = mid_n - mid_k
        end = mid_n + mid_k

        final = final_df.loc[start:end]
        final_image_path = final['image_path'].head(limit_image_to_train).tolist()
    else:
        print("get data from top row")
        final_image_path = final_df['image_path'].head(limit_image_to_train).tolist()
    
    
    # clear zombies memory
    del [[final_df, df_analysis]]
    gc.collect()
    
    return final_image_path

def sliding_crop_and_select_one(img, stepSize=stSize, windowSize=winSize):
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

def sliding_crop(img, stepSize=stSize, windowSize=winSize):
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


# In[ ]:


def read_data_with_labels(filepath, class_names, training, limit=100):
   
    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath,class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat
        path_list = []
        class_list = []
        for img in tqdm(os.listdir(path)):  
            if ".DS_Store" != img:
                filpath = os.path.join(path,img)
#                 print(filpath, class_num)
                
                path_list.append(filpath)
                class_list.append(class_num)
                # image_label_list.append({filpath:class_num})
        
        n_samples = None
        if limit != "MAX":
            n_samples = limit
                    
        path_list, class_list = shuffle(path_list, class_list, n_samples=n_samples ,random_state=random.randint(123, 10000))
        
        image_list = image_list + path_list
        label_list = label_list + class_list
  
    # print(image_list, label_list)
    
    return image_list, label_list

def prep_stage(x, train=True):
    beta_contrast = 0.1
    
    if train:
        # x = enhance_image(x, beta_contrast)
        x = tfa.image.equalize(x)
        # x = custom_v3(x)
    else: 
        # x = enhance_image(x, beta_contrast)
        x = tfa.image.equalize(x)
        # x = custom_v3(x)
    return x

def post_stage(x):
    
    x = tf.image.resize(x, (IMG_H, IMG_W))
    # x = tf.image.resize_with_crop_or_pad(x, IMG_H, IMG_W)
    # normalize to the range -1,1
    x = tf.cast(x, tf.float32)
    x = (x - 127.5) / 127.5
    # normalize to the range 0-1
    # img /= 255.0
    return x

def extraction(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    # img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img, True)
    img = sliding_crop_and_select_one(img)
    img = post_stage(img)

    return img, label

def extraction_test(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    # img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img, False)
    # img = post_stage(img)
    
    img_list = sliding_crop(img)
    
    img = [post_stage(a) for a in img_list]

    return img, label


# In[ ]:


def checking_gen_disc(mode, g_model_inner, d_model_inner, g_filepath, d_filepath, test_data_path):
    print("Start Checking Reconstructed Image")
    g_model_inner.load_weights(g_filepath)
    d_model_inner.load_weights(d_filepath)
    
    normal_image = glob.glob(test_data_path+"/normal/*.png")[0]
    defect_image = glob.glob(test_data_path+"/defect/*.png")[0]
    paths = {
        "normal": normal_image,
        "defect": defect_image,
    }

    for i, v in paths.items():
        print(i,v)

        width=IMG_W
        height=IMG_H
        rows = 1
        cols = 3
        axes=[]
        fig = plt.figure()

        
        img, label = extraction(v, i)
       
        name_subplot = mode+'_original_'+i
        axes.append( fig.add_subplot(rows, cols, 1) )
        axes[-1].set_title('_original_')  
        plt.imshow(img.numpy().astype("int64"), alpha=1.0)
        plt.axis('off')

       
        img = tf.cast(img, tf.float64)
        img = (img - 127.5) / 127.5


        image = tf.reshape(img, (-1, IMG_H, IMG_W, IMG_C))
        reconstructed_images = g_model_inner.predict(image)
        reconstructed_images = tf.reshape(reconstructed_images, (IMG_H, IMG_W, IMG_C))
        reconstructed_images = reconstructed_images * 127 + 127

        name_subplot = mode+'_reconstructed_'+i
        axes.append( fig.add_subplot(rows, cols, 3) )
        axes[-1].set_title('_reconstructed_') 
        plt.imshow(reconstructed_images.numpy().astype("int64"), alpha=1.0)
        plt.axis('off')

        fig.tight_layout()    
        fig.savefig(mode+'_'+i+'.png')
        plt.show()
        plt.clf()


# In[ ]:


class Dataset:
    # This class will facilitate the creation of a few-shot dataset
    # from the Omniglot dataset that can be sampled from quickly while also
    # allowing to create new labels at the same time.
    def __init__(self, path_file, training=True, limit=100):
        # Download the tfrecord files containing the omniglot data and convert to a
        # dataset.
        self.data = {}
        
        class_names = ["normal"] if training else ["normal", "defect"]
        filenames, labels = read_data_with_labels(path_file, class_names, training, limit)
        
        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        self.ds = ds.shuffle(buffer_size=1024, seed=random.randint(123, 10000) )
        
        
        if training:
            for image, label in ds.map(extraction):
                image = image.numpy()
                label = str(label.numpy())
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(image)
            self.labels = list(self.data.keys())
            

    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, split=False
    ):
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, IMG_H, IMG_W, IMG_C))
        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, IMG_H, IMG_W, IMG_C))

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        
        if split:
            return dataset, test_images, test_labels
        return dataset
    
    def get_dataset(self, batch_size):
        ds = self.ds.map(extraction_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

import urllib3

urllib3.disable_warnings() # Disable SSL warnings that may happen during download.

## load dataset
train_dataset = Dataset(train_data_path, training=True, limit=LIMIT_TRAIN_IMAGES)

eval_dataset = Dataset(eval_data_path, training=False, limit=LIMIT_EVAL_IMAGES)
eval_ds = eval_dataset.get_dataset(1)


# In[ ]:


# _, axarr = plt.subplots(nrows=2, ncols=5, figsize=(20, 20))
# sample_keys = list(test_dataset.data.keys())
# # print(sample_keys)
# for a in range(2):
#     for b in range(5):
#         temp_image = test_dataset.data[sample_keys[a]][b]
#         temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
#         temp_image *= 255
#         temp_image = np.clip(temp_image, 0, 255).astype("uint8")
#         if b == 2:
#             axarr[a, b].set_title("Class : " + sample_keys[a])
#         axarr[a, b].imshow(temp_image)
#         axarr[a, b].xaxis.set_visible(False)
#         axarr[a, b].yaxis.set_visible(False)
# plt.show()


# In[ ]:


def calculate_a_score(out_g_model, out_d_model, images):
    reconstructed_images = out_g_model(images, training=False)

    feature_real, label_real  = out_d_model(images, training=False)
    # print(generated_images.shape)
    feature_fake, label_fake = out_d_model(reconstructed_images, training=False)

    # Loss 2: RECONSTRUCTION loss (L1)
    loss_rec = mae(images, reconstructed_images)

    loss_feat = multimse(feature_real, feature_fake)
    # print("loss_rec:", loss_rec, "loss_feat:", loss_feat)
    score = (anomaly_weight * loss_rec) + ((1-anomaly_weight) * loss_feat)
    return score, loss_rec, loss_feat


# In[ ]:


def conv_block_2nd(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_2nd(x, num_filters)
    return x


# In[ ]:


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
        
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name


    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = add([x, input_tensor], name='block_' + block_name + '_x4')
    x = Activation('relu', name='block_out_' + block_name + '_x4')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name

    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)
    
    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])
    
    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '_prj')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_prj_bn')(shortcut)

    x = add([x, shortcut], name='block_' + block_name)
    x = Activation('relu', name='block_out_' + block_name)(x)
    return x


def SEResNet50(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=225,
                                      min_size=160,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name="input_1")
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, name="input_1")
        else:
            img_input = input_tensor
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    # x = ZeroPadding2D(padding=(2, 2), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='conv1_pool')(x)
    # x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # x = Flatten()(x)
    # x = Dense(classes, activation='softmax', name='fc6')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='se-resnet50')
    return model  


# In[ ]:


def build_seresnet50_unet(input_shape):
    inputs = Input(input_shape, name="input_1")
    """ Pre-trained ResNet50 Model """
    seresnet50 = SEResNet50(weights=None, input_tensor=inputs)
    seresnet50.summary()
    """ Encoder """
    s1 = seresnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = seresnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = seresnet50.get_layer("relu3_1_x1").output  ## (128 x 128)
    s4 = seresnet50.get_layer("relu4_1_x1").output  ## (64 x 64)
    s5 = seresnet50.get_layer("relu5_1_x1").output  ## (32 x 32)

    """ Bridge """
    b1 = seresnet50.get_layer("block_out_5_3_x4").output  ## (16 x 16)

    """ Decoder """
    x = IMG_H
    d1 = decoder_block(b1, s5, x)                     ## (32 x 32)
    x = x/2
    d2 = decoder_block(d1, s4, x)                     ## (64 x 64)
    x = x/2
    d3 = decoder_block(d2, s3, x)                     ## (128 x 128)
    x = x/2
    d4 = decoder_block(d3, s2, x)                      ## (256 x 256)
    x = x/2
    d5 = decoder_block(d4, s1, x)                      ## (512 x 512)
    
    """ Output """
    outputs = tf.keras.layers.Conv2D(IMG_C, 1, padding="same", activation="tanh")(d5)

    model = tf.keras.models.Model(inputs, outputs)

    return model


# In[ ]:


# create discriminator model
def build_discriminator(inputs):
    num_layers = 4
    if IMG_H > 128:
        num_layers = 5
    f = [2**i for i in range(num_layers)]
    x = inputs
    features = []
    for i in range(0, num_layers):
        if i == 0:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * IMG_H ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        else:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * IMG_H ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        
        features.append(x)
           
    x = tf.keras.layers.Flatten()(x)
    features.append(x)
    output = tf.keras.layers.Dense(1, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs = [features, output])
    
    return model


# In[ ]:


def testing(g_model_inner, d_model_inner, g_filepath, d_filepath, test_ds):
    class_names = ["normal", "defect"] # normal = 0, defect = 1
    
    g_model_inner.load_weights(g_filepath)
    d_model_inner.load_weights(d_filepath)
    
        
    scores_ano = []
    real_label = []
    rec_loss_list = []
    feat_loss_list = []
    ssim_loss_list = []
    counter = 0
    
    for images, labels in test_ds:
        loss_rec, loss_feat = 0.0, 0.0
        score = 0
        
        counter += 1
        '''for normal'''
        # temp_score, loss_rec, loss_feat = calculate_a_score(g_model_inner, d_model_inner, images)
        # score = temp_score.numpy()
        
        
        '''for sliding images & Crop LR'''
        for image in images:
            r_score, r_rec_loss, r_feat_loss = calculate_a_score(g_model_inner, d_model_inner, image)
            if r_score.numpy() > score or score == 0:
                score = r_score.numpy()
                loss_rec = r_rec_loss
                loss_feat = r_feat_loss
                
            
        scores_ano = np.append(scores_ano, score)
        real_label = np.append(real_label, labels.numpy()[0])
        
        rec_loss_list = np.append(rec_loss_list, loss_rec)
        feat_loss_list = np.append(feat_loss_list, loss_feat)
        if (counter % 100) == 0:
            print(counter, " tested.")
    ''' Scale scores vector between [0, 1]'''
    scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
    
    auc_out, threshold = roc(real_label, scores_ano, name_model)
    print("auc: ", auc_out)
    print("threshold: ", threshold)

    plot_anomaly_score(anomaly_scores, name, model_name)
    
    scores_ano = (scores_ano > threshold).astype(int)
    cm = tf.math.confusion_matrix(labels=real_label, predictions=scores_ano).numpy()
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    print(cm)
    print(
            "model saved. TP %d:, FP=%d, FN=%d, TN=%d" % (TP, FP, FN, TN)
    )
    plot_confusion_matrix(cm, class_names, title=name_model)

    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()

    print("Accuracy: ", diagonal_sum / sum_of_all_elements )
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rat (FNR): ", FN/(FN+TP))
    print("TNR: ", TN/(FP+TN))
    print("precision_score: ", TP/(TP+FP))
    print("recall_score: ", TP/(TP+FN))
    print("NPV: ", TN/(FN+TN))
    print("F1-Score: ", f1_score(real_label, scores_ano))


# In[ ]:


input_shape = (IMG_H, IMG_W, IMG_C)
# set input 
inputs = tf.keras.layers.Input(input_shape, name="input_1")
# inputs_disc = tf.keras.layers.Input((IMG_H, IMG_W, 1), name="input_1")

g_model = build_seresnet50_unet(input_shape)
d_model = build_discriminator(inputs)
# grayscale_converter = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))
d_model.compile()
g_model.compile()
# g_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)

# d_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)


# In[ ]:


ADV_REG_RATE_LF = 1
REC_REG_RATE_LF = 50
# SSIM_REG_RATE_LF = 10
FEAT_REG_RATE_LF = 1


gen_loss_list = []
disc_loss_list = []
iter_list = []
auc_list = []


# In[ ]:


@tf.function
def train_step(real_images):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # tf.print("Images: ", images)
        reconstructed_images = g_model(real_images, training=True)
        
        # real_images = grayscale_converter(real_images)
        feature_real, label_real = d_model(real_images, training=True)
        # print(generated_images.shape)
        feature_fake, label_fake = d_model(reconstructed_images, training=True)

        discriminator_fake_average_out = tf.math.reduce_mean(label_fake, axis=0)
        discriminator_real_average_out = tf.math.reduce_mean(label_real, axis=0)
        real_fake_ra_out = label_real - discriminator_fake_average_out
        fake_real_ra_out = label_fake - discriminator_real_average_out
        epsilon = 0.000001
        
        # Loss 1: 
        # use relativistic average loss
        loss_gen_ra = -( 
            tf.math.reduce_mean( 
                tf.math.log( 
                    tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0 
            ) + tf.math.reduce_mean( 
                tf.math.log(1-tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0 
            ) 
        )

        loss_disc_ra = -( 
            tf.math.reduce_mean( 
                tf.math.log(
                    tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0 
            ) + tf.math.reduce_mean( 
                tf.math.log(1-tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0 
            ) 
        )

        # Loss 2: RECONSTRUCTION loss (L1)
        loss_rec = mae(real_images, reconstructed_images)

        # Loss 3: SSIM Loss
        # # loss_ssim =  ssim(real_images, reconstructed_images)

        # Loss 4: FEATURE Loss
        # loss_feat = mse(feature_real, feature_fake)
        loss_feat = multimse(feature_real, feature_fake, FEAT_REG_RATE_LF)


        gen_loss = tf.reduce_mean( 
            (loss_gen_ra * ADV_REG_RATE_LF) 
            + (loss_rec * REC_REG_RATE_LF) 
            # + (loss_ssim * SSIM_REG_RATE_LF) 
            + (loss_feat) 
        )

        disc_loss = tf.reduce_mean( (loss_disc_ra * ADV_REG_RATE_LF) + (loss_feat * FEAT_REG_RATE_LF) )

    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)

    d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    g_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    
    return gen_loss, disc_loss


# In[ ]:


if TRAIN:
    print("Start Trainning. ", name_model)
    best_auc = 0.7
    
    start_time = datetime.now()
    for meta_iter in range(meta_iters):
        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        # Temporarily save the weights from the model.
        d_old_vars = d_model.get_weights()
        g_old_vars = g_model.get_weights()
        # Get a sample from the full dataset.
        mini_dataset = train_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots, classes
        )
        gen_loss_out = 0.0
        disc_loss_out = 0.0
        
        # print("meta_iter: ", meta_iter)
        for images, _ in mini_dataset:
            g_loss, d_loss = train_step(images)
            gen_loss_out = g_loss
            disc_loss_out = d_loss
            
        d_new_vars = d_model.get_weights()
        g_new_vars = g_model.get_weights()

        # Perform SGD for the meta step.
        for var in range(len(d_new_vars)):
            d_new_vars[var] = d_old_vars[var] + (
                (d_new_vars[var] - d_old_vars[var]) * cur_meta_step_size
            )

        for var in range(len(g_new_vars)):
            g_new_vars[var] = g_old_vars[var] + (
                (g_new_vars[var] - g_old_vars[var]) * cur_meta_step_size
            )

        # After the meta-learning step, reload the newly-trained weights into the model.
        g_model.set_weights(g_new_vars)
        d_model.set_weights(d_new_vars)
        
        # Evaluation loop
        meta_iter = meta_iter + 1
        if meta_iter % 100 == 0:
            eval_g_model = g_model
            eval_d_model = d_model
            
            iter_list = np.append(iter_list, meta_iter)
            gen_loss_list = np.append(gen_loss_list, gen_loss_out)
            disc_loss_list = np.append(disc_loss_list, disc_loss_out)

            scores_ano = []
            real_label = []
            counter = 0
           
            for images, labels in eval_ds:

                loss_rec, loss_feat = 0.0, 0.0
                score = 0
                counter += 1

                '''for normal'''
                # temp_score, loss_rec, loss_feat = calculate_a_score(eval_g_model, eval_d_model, images)
                # score = temp_score.numpy()


                '''for Sliding Images & LR Crop'''
                for image in images:
                    r_score, r_rec_loss, r_feat_loss = calculate_a_score(eval_g_model, eval_d_model, image)
                    if r_score.numpy() > score or score == 0:
                        score = r_score.numpy()
                        loss_rec = r_rec_loss
                        loss_feat = r_feat_loss
                    
                scores_ano = np.append(scores_ano, score)
                real_label = np.append(real_label, labels.numpy()[0])
                if (counter % 100) == 0:
                    print(counter, " tested.")
            # print("scores_ano:", scores_ano)
            '''Scale scores vector between [0, 1]'''
            scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
            # print("real_label:", real_label)
            # print("scores_ano:", scores_ano)
            auc_out, threshold = roc(real_label, scores_ano, name_model)
            auc_list = np.append(auc_list, auc_out)
            scores_ano = (scores_ano > threshold).astype(int)
            cm = tf.math.confusion_matrix(labels=real_label, predictions=scores_ano).numpy()
            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[0][0]
            # print(cm)
            print(
                f"model saved. batch {meta_iter}:, AUC={auc_out:.3f}, TP={TP}, TN={TN}, FP={FP}, FN={FN}, Gen Loss={gen_loss_out:.5f}, Disc Loss={disc_loss_out:.5f}" 
            )
            
            if auc_out >= best_auc:
                print(
                    f"the best model saved. at batch {meta_iter}: with AUC={auc_out:.3f}"
                )
                
                best_g_model_path = g_model_path.replace(".h5", f"_best_{meta_iter}_{auc_out:.2f}.h5")
                best_d_model_path = d_model_path.replace(".h5", f"_best_{meta_iter}_{auc_out:.2f}.h5")
                g_model.save(best_g_model_path)
                d_model.save(best_d_model_path)
                best_auc = auc_out
            # save model's weights
            g_model.save(g_model_path)
            d_model.save(d_model_path)
    
    
    end_time = datetime.now()
    print(f'Duration of Training: {end_time - start_time}')
    """
    Train Ends
    """
    plot_epoch_result(iter_list, gen_loss_list, "Generator_Loss", name_model, "g")
    plot_epoch_result(iter_list, disc_loss_list, "Discriminator_Loss", name_model, "r")
    plot_epoch_result(iter_list, auc_list, "AUC", name_model, "b")


# In[ ]:


test_dataset = Dataset(test_data_path, training=False, limit=LIMIT_TEST_IMAGES)

start_time = datetime.now()
testing(g_model, d_model, g_model_path, d_model_path, test_dataset.get_dataset(1))
end_time = datetime.now()
print(f'Duration of Testing: {end_time - start_time}')


# In[ ]:


checking_gen_disc(name_model, g_model, d_model, g_model_path, d_model_path, test_data_path)

