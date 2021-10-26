import coverage
import tensorflow as tf

cov = coverage.Coverage()
cov.start()

# # import tensorflow as tf

# # load model
# model = tf.keras.models.load_model('./alexnet-cifar10_origin.h5')

# # load datasets
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # predictions from original model
# predictions = model.predict(x_train)
import dummy
dummy.dummy()

cov.stop()

f = open('test.txt', 'w')
    
print(cov.report(file=f))