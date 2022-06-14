# reference https://www.geeksforgeeks.org/python-image-classification-using-keras/

import os
import random

import numpy as np
from keras.models import load_model
from keras.utils import load_img
from glob import glob

model_name = "save1655174607"
img_width, img_height = (290, 325)

base_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(
    base_path,
    "models",
    model_name
)

model = load_model(model_path)

file_names = glob(os.path.join(base_path, "data", "train", "*", "*.png"))
image_names = random.sample(file_names, int(0.1*len(file_names)))

for test_image in image_names:
    image = load_img(test_image, target_size=(img_width, img_height))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, img_width, img_height, 3)
    label = model.predict(img)
    round_label = np.around(label)

    print("Predicted Class: ", round_label, label)
