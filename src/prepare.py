import os
import numpy as np
from tensorflow import keras 

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
#this dataset already has a train test split

os.makedirs("data/raw",exist_ok=True)
np.savez_compressed("data/raw/fashion_raw.npz",
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)

print("Data:", x_train.shape, x_test.shape)
#60,000 images training, 10,000 for testing