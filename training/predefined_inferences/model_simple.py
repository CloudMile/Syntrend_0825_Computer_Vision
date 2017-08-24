
import tensorflow as tf

def inference(images, num_classes, dropout_keep_prob=0.6, is_training=True, reuse=False):

    # Convolutional Layer #1
    # Computes 16 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 200, 200, 3]
    # Output Tensor Shape: [batch_size, 200, 200, 16]
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=images,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu, 
            name='conv1')

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 200, 200, 16]
    # Output Tensor Shape: [batch_size, 100, 100, 16]
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same', 
            name='pool1')

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 16]
    # Output Tensor Shape: [batch_size, 100, 100, 16]
    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu, 
            name='conv2')

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 16]
    # Output Tensor Shape: [batch_size, 50, 50, 16]
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same', 
            name='pool2')

    # Dense Layer
    with tf.variable_scope('dense') as scope:
        b, x, y, n = pool2.shape
        dim = x.value * y.value * n.value
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 50, 50, 16]
        # Output Tensor Shape: [batch_size, 50 * 50 * 16]
        pool2_flat = tf.reshape(pool2, [-1, dim])
        # Densely connected layer with 128 neurons
        # Input Tensor Shape: [batch_size, 50 * 50 * 16]
        # Output Tensor Shape: [batch_size, 128]
        dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu, 
            name='dense')

    # Dropout Layer
    # Add dropout operation; 0.6 probability that element will be kept
    with tf.variable_scope('dropout') as scope:
        dropout = tf.layers.dropout( inputs=dense, rate=1 - dropout_keep_prob, training=is_training, 
            name='dropout')

    # Logits layer
    # Input Tensor Shape: [batch_size, 128]
    # Output Tensor Shape: [batch_size, num_classes]
    with tf.variable_scope('logits') as scope:
        logits = tf.layers.dense(inputs=dropout, units=num_classes, 
            name='logits')

    end_points = {
        # Generate predictions of class
        'Classes': tf.argmax(input=logits, axis=1, name='Classes'),
        # Generate possibilities for each classes 
        'Predictions': tf.nn.softmax(logits, name='Predictions')
    }
    return logits, end_points




