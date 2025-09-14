# ===================== WITHOUT KERAS =====================
# Full control, manual weights, manual forward/backward pass
# Good for understanding inner workings of neural networks

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# -------------------- Load & preprocess data --------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)   # Flatten images
x_test = x_test.reshape(-1, 784)

# One-hot encode labels
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# -------------------- Define model manually --------------------
# Input layer -> Hidden layer -> Output layer
W1 = tf.Variable(tf.random.normal([784, 128]))
b1 = tf.Variable(tf.zeros([128]))
W2 = tf.Variable(tf.random.normal([128, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Optimizer
optimizer = tf.optimizers.Adam()

# -------------------- Training step --------------------
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        hidden = tf.nn.relu(tf.matmul(x, W1) + b1)
        logits = tf.matmul(hidden, W2) + b2
        # Loss calculation
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    # Compute gradients
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    # Update weights
    optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))
    return loss

# Note: To train, you would loop over batches and call train_step manually
# e.g., for epoch in range(5): train_step(x_train_batch, y_train_batch)
