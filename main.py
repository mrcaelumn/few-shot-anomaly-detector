#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import itertools

import os
from glob import glob
from tqdm import tqdm
import numpy as np
import random
import gc
import pandas as pd 

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
import math
import natsort


# In[ ]:


from models.resnet50 import build_generator_resnet50_unet
from models.seresnet50 import build_seresnet50_unet
from models.seresnext50 import build_seresnext50_unet
from models.discriminator import build_discriminator


from models.custom_optimizers import GCAdam
from models.loss_func import SSIMLoss, AdversarialLoss, MultiFeatureLoss
from models.data_augmentation import selecting_images_preprocessing, sliding_crop     , sliding_crop_and_select_one, custom_v3, enhance_image


# In[ ]:


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-dn", "--DATASET_NAME", default="mura", help="name of dataset in data directory.")
parser.add_argument("-s", "--SHOTS", default=20, type=int, help="how many data that you want to use.")
parser.add_argument("-nd", "--NO_DATASET", default=0, type=int, help="select which number of dataset.")
parser.add_argument("-bb", "--BACKBONE", default="seresnet50", help="backbone model for generator's encoder. (resnet50, seresnet50, seresnext50)")
args = vars(parser.parse_args())


# In[ ]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.
winSize = (256, 256)
stSize = 20

# Weight initializers for the Generator network
# WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)

AUTOTUNE = tf.data.AUTOTUNE

LIMIT_EVAL_IMAGES = 100
LIMIT_TEST_IMAGES = 3000
LIMIT_TRAIN_IMAGES = 100

TRAINING_DURATION = None
TESTING_DURATION = None

NUMBER_IMAGES_SELECTED = 1000

# range between 0-1
anomaly_weight = 0.1

learning_rate = 0.002
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
inner_iters = 4


train_shots = 100
# shots = 20
shots = args["SHOTS"]
classes = 1
n_shots = shots
if shots > 20 :
    n_shots = "few"
    
# DATASET_NAME = "mura"
DATASET_NAME = args["DATASET_NAME"]
# NO_DATASET = 0 # 0=0-999 images, 1=1000-1999, 2=2000-2999 so on
NO_DATASET = args["NO_DATASET"] # 0=0-999 images, 1=1000-1999, 2=2000-2999 so on

PERCENTAGE_COMPOSITION_DATASET = {
    "top": 50,
    "mid": 40,
    "bottom": 10
}

mode_colour = str(IMG_H) + "_rgb"
if IMG_C == 1:
    mode_colour = str(IMG_H) + "_gray"

MODEL_BACKBONE = args["BACKBONE"]
name_model = f"{mode_colour}_{DATASET_NAME}_{NO_DATASET}_{MODEL_BACKBONE}_{n_shots}_shots_mura_detection_{str(meta_iters)}"
g_model_path = f"saved_model/{name_model}_g_model.h5"
d_model_path = f"saved_model/{name_model}_d_model.h5"
plot_folder = "plot_output/"
text_folder = "text_output/"

TRAIN = True
if not TRAIN:
    g_model_path = "saved_model/g_model_name.h5"
    d_model_path = "saved_model/d_model_name.h5"
    
train_data_path = f"data/{DATASET_NAME}/train_data"
eval_data_path = f"data/{DATASET_NAME}/eval_data"
test_data_path = f"data/{DATASET_NAME}/test_data"


# In[ ]:


def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(plot_folder + name_model+'_roc_curve.png')
    plt.show()
    plt.clf()

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
    plot_roc_curve(fpr, tpr, name_model)
    
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
    plt.savefig(plot_folder + title+'_cm.png')
    plt.show()
    plt.clf()
    
def plot_epoch_result(iters, loss, name, model_name, colour):
    plt.plot(iters, loss, colour, label=name)
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_folder + model_name+ '_'+name+'_iters_result.png')
    plt.show()
    plt.clf()

def plot_anomaly_score(score_ano, labels, name, model_name):
    
    df = pd.DataFrame(
    {'predicts': score_ano,
     'label': labels
    })
    
    df_normal = df[df.label == 0]
    sns.distplot(df_normal['predicts'],  kde=False, label='normal')

    df_defect = df[df.label == 1]
    sns.distplot(df_defect['predicts'],  kde=False, label='defect')
    
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Number of samples')
    plt.legend(prop={'size': 12})
    plt.savefig(plot_folder + model_name+ '_'+name+'_anomay_scores_dist.png')
    plt.show()
    plt.clf()

def write_result(array_lines, name):
    with open(f'{text_folder}{name}.txt', 'w+') as f:
        f.write('\n'.join(array_lines))


# In[ ]:


def read_data_with_labels(filepath, class_names, training=True, limit=100):
   
    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath,class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat
        path_list = []
        class_list = []
        
        list_path = natsort.natsorted(os.listdir(path))
        
        if training:
            print("total number of dataset", len(list_path))

            newarr_list_path = np.array_split(list_path, math.ceil(len(list_path)/NUMBER_IMAGES_SELECTED))

            print("number of sub dataset", len(newarr_list_path))

            list_path = newarr_list_path[NO_DATASET]

            print("data taken from dataset", len(list_path))
        
        
        for img in tqdm(list_path, desc='selecting images'):  
            if ".DS_Store" != img:
                # print(img)
                filpath = os.path.join(path,img)
#                 print(filpath, class_num)
                
                path_list.append(filpath)
                class_list.append(class_num)
                # image_label_list.append({filpath:class_num})
        
        n_samples = None
        if limit != "MAX":
            n_samples = limit
        else: 
            n_samples = len(path_list)
            
        if training:
            ''' 
            selecting by attribute of image
            '''
            combined = np.transpose((path_list, class_list))
            # print(combined)
            path_list, class_list = selecting_images_preprocessing(combined, limit_image_to_train=n_samples, composition=PERCENTAGE_COMPOSITION_DATASET)
        
        else:
            ''' 
            random selecting
            '''
            path_list, class_list = shuffle(path_list, class_list, n_samples=n_samples ,random_state=random.randint(123, 10000))
        
        image_list = image_list + path_list
        label_list = label_list + class_list
  
    # print(image_list, label_list)
    
    return image_list, label_list

def prep_stage(x, train=True):
    beta_contrast = 0.1
    # enchance the brightness
    x = enhance_image(x, beta_contrast)
    # if train:
        # x = enhance_image(x, beta_contrast)
        # x = tfa.image.equalize(x)
        # x = custom_v3(x)
    # else: 
        # x = enhance_image(x, beta_contrast)
        # x = tfa.image.equalize(x)
        # x = custom_v3(x)
        
    return x

def post_stage(x):
    
    x = tf.image.resize(x, (IMG_H, IMG_W))
    # x = tf.image.resize_with_crop_or_pad(x, IMG_H, IMG_W)
    # normalize to the range -1,1
    # x = tf.cast(x, tf.float32)
    x = (x - 127.5) / 127.5
    # normalize to the range 0-1
    # img /= 255.0
    return x

def extraction(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    # print(image, label)
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = prep_stage(img, True)
    
    img = sliding_crop_and_select_one(img)
    img = post_stage(img)

    return img, label

def extraction_test(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
    
    normal_image = glob(test_data_path+"/normal/*.png")[0]
    defect_image = glob(test_data_path+"/defect/*.png")[0]
    paths = {
        "normal": normal_image,
        "defect": defect_image,
    }

    for i, v in paths.items():
        print(i,v)

        rows = 1
        cols = 3
        axes=[]
        fig = plt.figure()

        
        img, label = extraction(v, i)
       
        axes.append( fig.add_subplot(rows, cols, 1) )
        axes[-1].set_title('_original_')  
        
        img = np.clip(img.numpy(), 0, 1)
        
        plt.imshow(img.astype(np.uint8), alpha=1.0)
        plt.axis('off')

       
        img = tf.cast(img, tf.float64)
        img = (img - 127.5) / 127.5


        image = tf.reshape(img, (-1, IMG_H, IMG_W, IMG_C))
        reconstructed_images = g_model_inner.predict(image)
        reconstructed_images = tf.reshape(reconstructed_images, (IMG_H, IMG_W, IMG_C))
        reconstructed_images = reconstructed_images * 127 + 127
        axes.append( fig.add_subplot(rows, cols, 3) )
        axes[-1].set_title('_reconstructed_') 
        
        reconstructed_images = np.clip(reconstructed_images.numpy(), 0, 1)
        
        plt.imshow(reconstructed_images.astype(np.uint8), alpha=1.0)
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
        start_time = datetime.now()
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
            
        end_time = datetime.now()
        
        print('classes: ', class_names)
        print(f'(Loading Dataset and Preprocessing) Duration of counting std and mean of images: {end_time - start_time}')
        

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
        dataset = dataset.shuffle(100, seed=int(round(datetime.now().timestamp()))).batch(batch_size).repeat(repetitions)
        
        if split:
            return dataset, test_images, test_labels
        return dataset
    
    def get_dataset(self, batch_size):
        ds = self.ds.map(extraction_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = tf.data.Dataset.from_tensor_slices((images, labels))
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


input_shape = (IMG_H, IMG_W, IMG_C)
# set input 
inputs = tf.keras.layers.Input(input_shape, name="input_1")

g_model = build_seresnet50_unet(input_shape, IMG_H, IMG_C)
d_model = build_discriminator(inputs, IMG_H)

if args["BACKBONE"] == "resnet50":
    
    print("backbone selected: resnet50")
    g_model = build_generator_resnet50_unet(input_shape, IMG_H, IMG_C)
elif args["BACKBONE"] == "seresnext50":
    
    print("backbone selected: seresnext50")
    g_model = build_seresnext50_unet(input_shape, IMG_H, IMG_C)
else:
    
    print("backbone selected (default): seresnext50")
    g_model = build_seresnet50_unet(input_shape, IMG_H, IMG_C)
    
# d_model.summary()
# g_model.summary()

d_model.compile()
g_model.compile()

g_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
# g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)

d_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
# d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)


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


def testing(g_model_inner, d_model_inner, g_filepath, d_filepath, test_ds):
    class_names = ["normal", "defect"] # normal = 0, defect = 1
    start_time = datetime.now()
    g_model_inner.load_weights(g_filepath)
    d_model_inner.load_weights(d_filepath)
    
        
    scores_ano = []
    real_label = []
    rec_loss_list = []
    feat_loss_list = []
    # ssim_loss_list = []
    # counter = 0
    
    for images, labels in tqdm(test_ds, desc='testing stages'):
        loss_rec, loss_feat = 0.0, 0.0
        score = 0
        
        # counter += 1
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
        # if (counter % 100) == 0:
        #     print(counter, " tested.")
    ''' Scale scores vector between [0, 1]'''
    scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
    
    auc_out, threshold = roc(real_label, scores_ano, name_model)
    print("auc: ", auc_out)
    print("threshold: ", threshold)
    
    # histogram distribution of anomaly scores
    plot_anomaly_score(scores_ano, real_label, "anomaly_score_dist", name_model)
    
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
    
    end_time = datetime.now()
    TESTING_DURATION = end_time - start_time
    print(f'Duration of Testing: {end_time - start_time}')
    arr_result = [
        f"Model Spec: {name_model}",
        f"AUC: {auc_out}",
        f"Accuracy: {(diagonal_sum / sum_of_all_elements)}",
        f"False Alarm Rate (FPR): {(FP/(FP+TN))}", 
        f"TNR: {(TN/(FP+TN))}", 
        f"Precision Score (PPV): {(TP/(TP+FP))}", 
        f"Recall Score (TPR): {(TP/(TP+FN))}", 
        f"NPV: {(TN/(FN+TN))}", 
        f"F1-Score: {(f1_score(real_label, scores_ano))}", 
        f"Training Duration: {TRAINING_DURATION}",
        f"Testing Duration: {TESTING_DURATION}"
    ]
    print("\n".join(arr_result))
    
    # print("Accuracy: ", diagonal_sum / sum_of_all_elements)
    # print("False Alarm Rate (FPR): ", FP/(FP+TN))
    # print("Leakage Rat (FNR): ", FN/(FN+TP))
    # print("TNR: ", TN/(FP+TN))
    # print("precision_score: ", TP/(TP+FP))
    # print("recall_score: ", TP/(TP+FN))
    # print("NPV: ", TN/(FN+TN))
    # print("F1-Score: ", f1_score(real_label, scores_ano))
    
    write_result(arr_result, name_model)


# In[ ]:


ADV_REG_RATE_LF = 1
REC_REG_RATE_LF = 50
SSIM_REG_RATE_LF = 10
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
        loss_ssim =  ssim(real_images, reconstructed_images, IMG_H)

        # Loss 4: FEATURE Loss
        # loss_feat = mse(feature_real, feature_fake)
        loss_feat = multimse(feature_real, feature_fake, FEAT_REG_RATE_LF)

        gen_loss = tf.reduce_mean( 
            (loss_gen_ra * ADV_REG_RATE_LF) 
            + (loss_rec * REC_REG_RATE_LF) 
            + (loss_ssim * SSIM_REG_RATE_LF) 
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
    # for meta_iter in tqdm(range(meta_iters), desc=f'training process'):
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
            # counter = 0
           
            for images, labels in tqdm(eval_ds, desc=f'evalution stage at {meta_iter} batch'):

                loss_rec, loss_feat = 0.0, 0.0
                score = 0
                # counter += 1
                
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
                # if (counter % 100) == 0:
                #     print(counter, " tested.")
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
    TRAINING_DURATION = end_time - start_time
    print(f'Duration of Training: {end_time - start_time}')
    """
    Train Ends
    """
    plot_epoch_result(iter_list, gen_loss_list, "Generator_Loss", name_model, "g")
    plot_epoch_result(iter_list, disc_loss_list, "Discriminator_Loss", name_model, "r")
    plot_epoch_result(iter_list, auc_list, "AUC", name_model, "b")


# In[ ]:


test_dataset = Dataset(test_data_path, training=False, limit=LIMIT_TEST_IMAGES)
testing(g_model, d_model, g_model_path, d_model_path, test_dataset.get_dataset(1))


# In[ ]:


checking_gen_disc(name_model, g_model, d_model, g_model_path, d_model_path, test_data_path)


# In[ ]:


gc.collect()

