
import tensorflow as tf
import predefined_inferences.model_inception_v3 as Inception_v3
import predefined_inferences.model_resnet_v2 as Resnet_v2
import predefined_inferences.model_resnet_v1 as Resnet_v1
import predefined_inferences.model_cifar10 as Cifar10
import predefined_inferences.model_simple as Simple
import predefined_inferences.model_custom as Custom

models = ['Inception_v3', 'Resnet_v1', 'Resnet_v2', 'Cifar10', 'Simple', 'Custom']

def summaryWriter(path, sess):
    summary_writer = tf.train.SummaryWriter(path, graph_def=sess.graph_def)
    return summary_writer

def writeSummaries(acc, loss, scope):
    tf.summary.scalar(scope+".Accuracy", acc)
    tf.summary.scalar(scope+".Loss", loss)    

def classification_inference(inputs, labels, model_name, for_training=False, reuse=False, dropout=0.5):
    '''
    Args:
        inputs:         tensor type
        labels:         list type (1D array)
        model_name:     string type
        for_training:   bool type
        reuse:          bool type
        dropout:        float type
    Returns:
        logits:         tensor type
        pred:           tensor type
        classes:        tensor type (vairable)
    '''
    if model_name not in models:
        raise ValueError('Unknown model name: ', model_name)

    print('Using inference: ', model_name, ' with classes: ', labels)
    n_classes = len(labels)
    classes = tf.Variable(labels, name='Labels', trainable=False)
    logits = None
    pred = None
    dropout_keep = 1-dropout
    if model_name == 'Inception_v3':
        logits, endpoints = Inception_v3.inference(inputs, num_classes=n_classes, is_training=for_training, dropout_keep_prob=dropout_keep)
    if model_name == 'Resnet_v1':
        logits, endpoints = Resnet_v1.inference(inputs, num_classes=n_classes, is_training=for_training, dropout_keep_prob=dropout_keep)
    elif model_name == 'Resnet_v2':
        logits, endpoints = Resnet_v2.inference(inputs, num_classes=n_classes, is_training=for_training, dropout_keep_prob=dropout_keep)
    elif model_name == 'Cifar10':
        logits, endpoints = Cifar10.inference(inputs, num_classes=n_classes, is_training=for_training,dropout_keep_prob=dropout_keep)
    elif model_name == 'Custom':
        logits, endpoints = Custom.inference(inputs, num_classes=n_classes, is_training=for_training, dropout_keep_prob=dropout_keep)
    else:
        logits, endpoints = Simple.inference(inputs, num_classes=n_classes, is_training=for_training, dropout_keep_prob=dropout_keep)

    pred = endpoints['Predictions']        
    return logits, pred, classes

def losses(logits, labels):
    '''
    Args:
        logits: tensor type
        labels: list type (1D array)
    Returns:
        loss: tensor type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss

def training(loss, learning_rate):
    '''
    Args:
        loss: tensor type
        learning_rate: float type
    Returns:
        train_op: tensor type
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_op = optimizer.minimize(loss)
    return train_op

def evaluation(logits, labels):
    '''
    Args:
        logits: tensor type
        labels: 1D list type (1D array)
    Returns:
        accuracy: tensor type
    '''
    # Calculate the accuracy
    with tf.variable_scope('accuracy') as scope:
        logits = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_prediction = tf.equal(logits, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

