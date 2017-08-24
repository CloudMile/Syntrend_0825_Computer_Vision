import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import os
import math
from PIL import Image

def readImagesInDirectory(prediction_dir, format='jpg'):
    tmp_imgs = []
    for file in os.listdir(prediction_dir):
        if file.endswith(format):
            img = Image.open(os.path.join(prediction_dir, file))
            arr = np.array(img)
            tmp_imgs.append(arr)
    return tmp_imgs

def readFilesInLabeledDirectory(dir, ratio, format='jpg'):
    '''
    Args:
        file_dir: file directory
        ratio:    percentation of validation data
    Returns:
        list of images and labels
    '''
    labels = []
    imgs = []
    labelNames = []
    count = 0
    for subdir in os.listdir(dir):
        if (os.path.isdir(os.path.join(dir, subdir)) == True):
            tmp_label = []
            tmp_img = []
            subpath = os.path.join(dir, subdir)
            labelNames.append(subdir)
            for file in os.listdir(subpath):
                if file.endswith(format):
                    tmp_img.append(os.path.join(subpath, file))
                    tmp_label.append(count)
            labels.append(tmp_label)
            imgs.append(tmp_img)
            count += 1

    image_list = np.hstack((imgs))
    label_list = np.hstack((labels))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images,tra_labels,val_images,val_labels,labelNames

def image_random_flip(image):
    return tf.image.random_flip_left_right(image)

def image_random_color(image, color_ordering=0, fast_mode=True, scope=None):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return image

def image_random_shift(image, width, height):
    return tf.random_crop(image, [width, height, 3])

def get_distorted_files(image, label, image_W, image_H, batch_size, capacity=1000, distorting=False):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
    batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    with tf.variable_scope('input_slicing') as scope:
        # make an input queue
        # Implemented using a Queue -- a QueueRunner for the Queue is added to 
        # the current Graph's QUEUE_RUNNER collection.
        input_queue = tf.train.slice_input_producer([image, label])
        label = input_queue[1]

    with tf.variable_scope('image_loading') as scope:
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image_contents, channels=3)
        # resize image to desirable size
        image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    ######################################
    # data argumentation should go to here
    if distorting:
        with tf.variable_scope('image_argumentation') as scope:
            t_rand = tf.random_uniform([1], minval = 2, maxval=1000, dtype=tf.int32)
            image = tf.cond(tf.equal(t_rand[0] % 2, 0), lambda: image_random_flip(image),  lambda: image)
            image = tf.cond(tf.equal(t_rand[0] % 3, 0), lambda: image_random_color(image),  lambda: image)
            t_rand_s = tf.random_uniform([1], minval = 0, maxval=10, dtype=tf.int32)
            image = image_random_shift(image, image_W - t_rand_s[0], image_H - t_rand_s[0])
            # resize image to desired size after cropping
            image = tf.image.resize_images(image, [image_W, image_H])
            image = tf.reshape(image, tf.stack([image_W, image_H, 3]))

            # image = image_random_flip(image)
            # image = image_random_color(image)
            # image = image_random_shift(image, image_W - 20, image_H - 20)
    ###################################### 
    
    with tf.variable_scope('image_normalization') as scope:
        image = tf.image.per_image_standardization(image)

    with tf.variable_scope('image_batch') as scope:
        batch = tf.train.batch([image, label],
                                batch_size= batch_size,
                                capacity = capacity)

    # batch is a list of tensor containing image and label
    return batch
