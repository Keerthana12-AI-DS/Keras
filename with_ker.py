# ===================== WITH KERAS =====================
# High-level API, much shorter, automatically handles training
# Recommended for most projects and prototyping

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -------------------- Load & preprocess data --------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)   # Flatten images
x_test = x_test.reshape(-1, 784)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------- Build model --------------------
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer
    layers.Dense(10, activation='softmax')                     # Output layer
])

# -------------------- Compile model --------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- Train model --------------------
model.fit(x_train, y_train, epochs=5, batch_size=32)

# -------------------- Evaluate --------------------
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
