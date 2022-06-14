# reference https://www.geeksforgeeks.org/python-image-classification-using-keras/

import os
from keras.models import load_model
from keras.utils import load_img
import numpy as np

from keras.models import load_model

model_name = "save1655165289"
test_image_name = "1654905623.png"
img_width, img_height = (290, 325)

module_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(
    module_path,
    "models",
    model_name
)

model = load_model(model_path)


test_data_dir = os.path.join(module_path, 'data', 'train', 'C3')
test_image = os.path.join(test_data_dir, test_image_name)



image = load_img(test_image, target_size=(img_width, img_height))
img = np.array(image)
img = img / 255.0
img = img.reshape(1, img_width, img_height, 3)
label = model.predict(img)
round_label = np.around(label)

print("Predicted Class: ", round_label, label)
