import gzip, pickle, os.path
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 10
WEIGHT_DECAY = 0.00002


# This builds the feedforward network op and returns it. A weight
# decay term is added to a collection so it can be referred to by the
# loss function.
def inference(images, name='m0', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        w1 = tf.get_variable("w1", (IMAGE_PIXELS, HIDDEN_SIZE),
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [HIDDEN_SIZE],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", (HIDDEN_SIZE, NUM_CLASSES),
                             initializer=tf.contrib.layers.xavier_initializer())
        l1 = tf.nn.relu(tf.matmul(images, w1) + b1)
        l2 = tf.matmul(l1, w2)
        # add weight decay to 'losses" collection
        weight_decay = tf.multiply(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2),
                                   WEIGHT_DECAY, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return l2


# The loss op. Take average cross-entropy loss over all of the
# examples in the batch and add the weight decay term.
def loss(logits, y):
    labels = tf.cast(y, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # This is just the single weight decay term in this case
    weight_reg = tf.add_n(tf.get_collection('losses'))

    # The loss is defined as the cross entropy loss plus the weight
    # decay term (L2 loss).
    loss = cross_entropy_mean + weight_reg

    return loss


# The training op.
def training(loss, x, learning_rate, decay_step, decay_factor):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_step,
                                    decay_factor, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)
    return train_op


# Run the network forward to produce predicted labels.
def predictions(logits):
    return tf.cast(tf.argmax(logits, axis=1), tf.int32)


def save_weights(sess, dir='models'):
    os.makedirs(dir, exist_ok=True)
    all_vars = tf.trainable_variables()
    with gzip.open(dir + "/mnist_params.pkl.gz", "w") as f:
        pickle.dump(tuple(map(lambda x: x.eval(sess), all_vars)), f)


def load_weights(sess, dir, model_name='m0'):
    i = 0
    filename = dir + '/mnist_params.pkl.gz' if dir else 'mnist_params.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
        w1, b1, w2 = tuple(weights[i:i+3])
        i += 3
        with tf.variable_scope(model_name, reuse=True):
            w1_var = tf.get_variable("w1", (IMAGE_PIXELS, HIDDEN_SIZE))
            b1_var = tf.get_variable("b1", (HIDDEN_SIZE))
            w2_var = tf.get_variable("w2", (HIDDEN_SIZE, NUM_CLASSES))
            sess.run([w1_var.assign(w1), b1_var.assign(b1),
                      w2_var.assign(w2)])
