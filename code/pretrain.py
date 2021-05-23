import numpy as np
import os
import cv2
import gt_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import ground_truth
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import load_model
model = load_model('./model/model_hand.h5')

print(model.summary())
print(model.optimizer.get_config())

path = "./data/"
files = os.listdir(path)
sorted_files = sorted(files,key=lambda x: int(x.split(".")[0]))

x = np.array([cv2.imread(path+file, 0) for file in sorted_files])
x = x.reshape(x.shape[0], x.shape[1], x.shape[2],1)

y = gt_dataset.data
y = to_categorical(y, num_classes=26, dtype='int')

x, y = shuffle(x, y, random_state=0)
x = x[:]
y = y[:]

print("Shape of train data: ", x.shape)
print("Shape of train labels: ", y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

history = model.fit(x_train, y_train, epochs=10,  validation_data = (x_test,y_test), verbose=1)
#model.save("./model/model_print1900.h5")




