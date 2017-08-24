
import tensorflow as tf

def inference(images, num_classes, dropout_keep_prob=0.6, is_training=True, reuse=False):
    # Convolutional Layer #1
    # Computes 16 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 200, 200, 3]
    # Output Tensor Shape: [batch_size, 200, 200, 16]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)

    #pool1 and norm1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
        #                   beta=0.75,name='norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')


    #pool2 and norm2
    with tf.variable_scope('pool2') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
        #                   beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling2')

    #local3
    with tf.variable_scope('local3') as scope:
        b, x, y, n = pool2.shape
        dim = x.value * y.value * n.value
        reshape = tf.reshape(pool2, shape=[-1, dim])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


    # Add dropout operation; 0.7 probability that element will be kept
    with tf.variable_scope('dropout') as scope:
        dropout = tf.layers.dropout( inputs=local3, rate=1 - dropout_keep_prob, training=is_training)

    # #local4
    # with tf.variable_scope('local4') as scope:
    #     weights = tf.get_variable('weights',
    #                               shape=[128,128],
    #                               dtype=tf.float32,
    #                               initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
    #     biases = tf.get_variable('biases',
    #                              shape=[128],
    #                              dtype=tf.float32,
    #                              initializer=tf.constant_initializer(0.1))
    #     local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('logits') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, num_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(dropout, weights), biases, name='logits')


    end_points = {
        'Classes': tf.argmax(input=logits, axis=1, name='Classes'),
        'Predictions': tf.nn.softmax(logits, name='Predictions')
    }
    return logits, end_points


