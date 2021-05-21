import numpy as np
import os
import cv2
import gt_dataset
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
print("Shape of train data: ", x.shape)

y = gt_dataset.data
y = to_categorical(y, num_classes = 26, dtype='int')
print("Shape of train labels: ", y.shape)

history = model.fit(x, y, epochs=10,  validation_data = (x,y), verbose=1)
model.save("./model/model_print1300.h5")

"""N = 9
pred = model.predict(x[:N])
for i in range(N):
    res = gt_dataset.word_dict[np.argmax(pred[i])]
    print(res)"""


