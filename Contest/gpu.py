import tensorflow as tf
import time

print("GPUs:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    a = tf.random.normal([8000, 8000])
    b = tf.random.normal([8000, 8000])

start = time.time()
c = tf.matmul(a, b)
print("Time:", time.time() - start)