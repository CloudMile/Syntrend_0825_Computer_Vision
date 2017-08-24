
import tensorflow as tf

def inference(images, num_classes, is_training=True, reuse=False):

    # Convolutional Layer #1
    # Computes 64 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 200, 200, 3]
    # Output Tensor Shape: [batch_size, 200, 200, 64]
    conv1 = tf.layers.conv2d(
        inputs=images,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 3x3 filter and stride of 2
    # Input Tensor Shape: [batch_size, 200, 200, 64]
    # Output Tensor Shape: [batch_size, 100, 100, 64]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding='same')

    # Local Response Normalization Layer #1
    # Input Tensor Shape: [batch_size, 100, 100, 64]
    # Output Tensor Shape: [batch_size, 100, 100, 64]
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 64]
    # Output Tensor Shape: [batch_size, 100, 100, 64]
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Local Response Normalization Layer #2
    # Input Tensor Shape: [batch_size, 100, 100, 64]
    # Output Tensor Shape: [batch_size, 100, 100, 64]
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

    # Pooling Layer #2
    # Second max pooling layer with a 3x3 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 64]
    # Output Tensor Shape: [batch_size, 50, 50, 64]
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3, 3], strides=2, padding='same')

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 50, 50, 64]
    # Output Tensor Shape: [batch_size, 50 * 50 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 50 * 50 * 64])

    # Local Layer #3
    # Densely connected layer with 384 neurons
    # Input Tensor Shape: [batch_size, 50 * 50 * 32]
    # Output Tensor Shape: [batch_size, 384]
    fc3 = tf.layers.dense(inputs=pool2_flat, units=384, activation=tf.nn.relu)

    # Local Layer #4
    # Densely connected layer with 384 neurons
    # Input Tensor Shape: [batch_size, 384]
    # Output Tensor Shape: [batch_size, 192]
    fc4 = tf.layers.dense(inputs=fc3, units=192, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 192]
    # Output Tensor Shape: [batch_size, num_classes]
    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    end_points = {
        'Classes': tf.argmax(input=logits, axis=1, name='Classes'),
        'Predictions': tf.nn.softmax(logits, name='Predictions')
    }
    return logits, end_points
