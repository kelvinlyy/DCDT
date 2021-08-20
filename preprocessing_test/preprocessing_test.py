import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import os

# load model
model = tf.keras.models.load_model('./alexnet-cifar10_origin.h5')

# load datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# assert datasets
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])


checkpoint_path = "training_1/cp.ckpt"

model.load_weights(checkpoint_path)


history = model.fit(x_train[:45000], y_train[:45000], batch_size=64, epochs=5, validation_data=(x_train[45000:], y_train[45000:]))

model.save_weights(checkpoint_path)
predictions = model.predict(x_test)

labels = ["airplane", "automobile", "bird", "cat", "deer", 'dog', "frog", "horse", "ship", "truck"]

plt.figure(figsize=(30, 30))
for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(x_test[i])
    plt.title(labels[np.argmax(predictions[i])])
    plt.axis("off")

plt.savefig('predictions.png')
