import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy

from pycbml.file_utils import prepare_files


def eval_and_test_model(img_width, img_height, base_path, model_name, overwrite=False):
    data_dir, model_save_path = prepare_files(base_path, overwrite=overwrite)
    
    model_path = os.path.join(model_save_path, model_name)
    test_data_dir = os.path.join(data_dir, "test")

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
    eval_and_test_model(img_width, img_height, base_path, model_name)
