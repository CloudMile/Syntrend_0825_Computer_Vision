from PIL import Image
import tensorflow as tf
import numpy as np
import os
import model

def restore_model(model_name, log_dir, sess, t_input, labels):
    print('Restoring mode ' + model_name + ' from ' + log_dir + ' with labels: ', labels)
    logits, op_pred, classes = model.classification_inference(t_input, labels, model_name, for_training=False, reuse=True)
    latest_ckpt = tf.train.latest_checkpoint(log_dir)
    saver = tf.train.Saver()
    # Or create saver from a meta file 
    # new_saver = tf.train.import_meta_graph('model.ckpt-600.meta')
    saver.restore(sess=sess, save_path=latest_ckpt)
    labels = sess.run(classes)
    return op_pred, labels

def classify_images(model_name, model_dir, cropped_images, labels, image_size=200):
    cropped_images = np.asarray(cropped_images)
    results = []
    tf.reset_default_graph()
    with tf.Session() as sess:
        print(len(cropped_images), image_size)
        t_images = tf.image.resize_image_with_crop_or_pad(cropped_images, image_size, image_size)
        t_images = tf.cast(t_images, tf.float32)
        t_images = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), t_images)
        op_pred, labels = restore_model(model_name, model_dir, sess, t_images, labels)
        results = sess.run(op_pred)
    return results, labels


# Classify images using placeholder
# def classify_images(model_name, model_dir, cropped_images, labels, image_size=200):
#     results = []
#     tf.reset_default_graph()
#     with tf.Session() as sess:
#         x = tf.placeholder(tf.float32, shape=[image_size, image_size, 3])
#         image = tf.image.resize_image_with_crop_or_pad(x, 200, 200)
#         image = tf.image.per_image_standardization(image)
#         image = tf.cast(image, tf.float32)
#         image = tf.reshape(image, [1, image_size, image_size, 3])
#         op_pred, labels =restore_model(model_name, model_dir, sess, image, labels)
#         for idx, image_rgb in enumerate(cropped_images):
#             result = sess.run(op_pred, feed_dict={x:image_rgb})  
#             result = result.squeeze()
#             results.append(result)
#     return results, labels