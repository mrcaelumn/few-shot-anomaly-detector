#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


# In[ ]:


IMG_H = 64
IMG_W = 64
IMG_C = 3  ## Change this to 1 for grayscale.

# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)

latent_dim = 128
    
learning_rate = 0.00001
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 100000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 10
classes = 10


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


class Dataset:
    # This class will facilitate the creation of a few-shot dataset
    # from the Omniglot dataset that can be sampled from quickly while also
    # allowing to create new labels at the same time.
    def __init__(self, training):
        # Download the tfrecord files containing the omniglot data and convert to a
        # dataset.
        split = "train" if training else "test"
        ds = tfds.load("tf_flowers", split=split, as_supervised=True, shuffle_files=False)
        # Iterate over the dataset to get each individual image and its class,
        # and put that data into a dictionary.
        self.data = {}

        def extraction(image, label):
            # This function will shrink the Omniglot images to the desired size,
            # scale pixel values and convert the RGB image to grayscale
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [IMG_H, IMG_W])
            return image, label

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


import urllib3

urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)


# In[ ]:


_, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())

for a in range(5):
    for b in range(5):
        temp_image = train_dataset.data[sample_keys[a]][b]
        temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
        temp_image *= 255
        temp_image = np.clip(temp_image, 0, 255).astype("uint8")
        if b == 2:
            axarr[a, b].set_title("Class : " + sample_keys[a])
        axarr[a, b].imshow(temp_image, cmap="gray")
        axarr[a, b].xaxis.set_visible(False)
        axarr[a, b].yaxis.set_visible(False)
plt.show()


# In[ ]:


# create generator model based on resnet50 and unet network
def build_generator(input_shape):
    model = tf.keras.Sequential()
    
    # Random noise to 16x16x256 image
    # model.add(tf.keras.layers.Dense(1024, activation="relu", use_bias=False, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(4*4*512, input_shape=input_shape))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape([4,4,512]))
    
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (5,5),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(128, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
  
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5,5),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(64, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(32, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    model.add(tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(2,2),use_bias=False,padding="same",kernel_initializer=WEIGHT_INIT,
                                     activation="tanh"
                                    ))
              # Tanh activation function compress values between -1 and 1. 
              # This is why we compressed our images between -1 and 1 in readImage function.
    # assert model.output_shape == (None,128,128,3)
    
    return model


# In[ ]:


# create discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(512,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return model


# In[ ]:


# we'll use cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

def wasserstein_loss(real_images, fake_images):
	return tf.keras.backend.mean(real_images * fake_images)

def generator_loss(fake_output):
    # First argument of loss is real labels
    # We've labeled our images as 1 (real) because
    # we're trying to fool discriminator
    return cross_entropy(tf.ones_like(fake_output),fake_output)


def discriminator_loss(real_images,fake_images):
    real_loss = cross_entropy(tf.ones_like(real_images),real_images)
    fake_loss = cross_entropy(tf.zeros_like(fake_images),fake_images)
    total_loss = real_loss + fake_loss
    return total_loss


# In[ ]:


input_shape = (IMG_H, IMG_W, IMG_C)

d_model = build_discriminator(input_shape)
g_model = build_generator((latent_dim, ))
d_model.compile()
g_model.compile()

g_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
d_optimizer = GCAdam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)


# In[ ]:


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
    for images, labels in mini_dataset:
        
        noise = tf.random.normal([inner_batch_size, latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator generated images
            generated_images = g_model(noise, training=True)

            # We've sent our real and fake images to the discriminator
            # and taken the decisions of it.
            real_output = d_model(images,training=True)
            fake_output = d_model(generated_images,training=True)

            # We've computed losses of generator and discriminator
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output,fake_output)

        # We've computed gradients of networks and updated variables using those gradients.
        gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
        g_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
        
        
        
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
    if meta_iter % eval_interval == 0:
        mini_dataset = test_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots, classes
        )
        
        d_old_vars = d_model.get_weights()
        g_old_vars = g_model.get_weights()
        
        for images, labels in mini_dataset:
            noise = tf.random.normal([inner_batch_size, latent_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generator generated images
                generated_images = g_model(noise, training=True)

                # We've sent our real and fake images to the discriminator
                # and taken the decisions of it.
                real_output = d_model(images, training=True)
                fake_output = d_model(generated_images, training=True)

                # We've computed losses of generator and discriminator
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output,fake_output)

            # We've computed gradients of networks and updated variables using those gradients.
            gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

            g_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
            d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
        
        # g_model.set_weights(g_old_vars)
        # d_model.set_weights(d_old_vars)
        
        if meta_iter % 100 == 0:
            print(
                "generate image in batch %d:" % (meta_iter)
            )
            noise = np.random.normal(size=(inner_batch_size, latent_dim))
            examples = g_model.predict(noise)
            save_plot(examples, meta_iter, int(np.sqrt(inner_batch_size)))
    
    


# In[ ]:


noise = tf.random.normal([inner_batch_size, latent_dim])
examples = g_model.predict(noise)
save_plot(examples, "few-shot-gan", int(np.sqrt(inner_batch_size)))
    # Train on the samples and get the resulting accuracies.


# In[ ]:




