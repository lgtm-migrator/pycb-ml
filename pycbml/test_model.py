import os
from glob import glob

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy


def eval_and_test_model(img_width, img_height, base_path, model_name):
    model_path = os.path.join(
    base_path,
    "models",
    model_name
    )
    test_data_dir = os.path.join(base_path,"data", "test")

    test_generator = ImageDataGenerator().flow_from_directory(
        test_data_dir,
        color_mode='rgb',
        target_size=(img_width, img_height),
        shuffle=False)

    model = load_model(model_path)

    score = model.evaluate(test_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predictions = model.predict(test_generator)
    pred_labels = numpy.argmax(predictions, axis=1)
    print(classification_report(pred_labels,test_generator.labels))

if __name__ == "__main__":
    model_name = "save1655262102"
    img_width, img_height = 290, 325

    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join("C:\\", "Google Drive",
                            "Programming", "Python", "pycb-ml")
    eval_and_test_model(img_width, img_height, base_path, model_name)
