import os
import random
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import set_random_seed
from numpy.random import seed

from data_statistics import get_class_stats, print_statistics

set_random_seed(2)
seed(1)


def copy_list_of_files(file_list, desitation_dir):
    for file_name in file_list:
        shutil.copy2(file_name, desitation_dir)


def copy_files_to_temp(classes, train_data_dir):
    temp_train_dir = tempfile.mkdtemp()
    temp_validate_dir = tempfile.mkdtemp()

    class_stats = get_class_stats(classes, train_data_dir)
    print("Input data stats:")
    print_statistics(class_stats)

    # get value to normalize training data count
    smallest_class_size = min(class_stats.values())

    for class_name in classes:
        class_data_dir = os.path.join(train_data_dir, class_name)

        # make dir for class in temp dirs
        os.makedirs(os.path.join(temp_train_dir, class_name))
        os.makedirs(os.path.join(temp_validate_dir, class_name))

        # get file list for class cut to smallest class size
        file_names = [os.path.join(class_data_dir, file_name) for file_name in next(
            os.walk(class_data_dir), (None, None, []))[2]]  # [:smallest_class_size]

        # calculate sample sizes
        validation_proportion = 0.2
        validate_sample_size = int(
            len(file_names) * validation_proportion)

        # create train and validate samples
        validate_sample = random.sample(
            file_names, validate_sample_size)
        train_sample = [
            x for x in file_names if x not in validate_sample]

        # copy samples into directories
        copy_list_of_files(validate_sample, os.path.join(
            temp_validate_dir, class_name))
        copy_list_of_files(train_sample, os.path.join(
            temp_train_dir, class_name))

    print("Normalized training data stats:")
    print_statistics(get_class_stats(classes, temp_train_dir))
    print("Normalized validation data stats:")
    print_statistics(get_class_stats(classes, temp_validate_dir))

    return temp_train_dir, temp_validate_dir


def prep_files_for_training(classes):
    module_path = os.path.dirname(os.path.realpath(__file__))

    train_data_dir = os.path.join(module_path, 'data', 'train')

    model_save_path = os.path.join(
        module_path,
        "models",
        f"save{str(int(time.time()))}"
    )

    temp_train_dir, temp_validate_dir = copy_files_to_temp(
        classes, train_data_dir)

    return temp_train_dir, temp_validate_dir, model_save_path


def compile_model(img_width, img_height):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = VGG16(weights='imagenet', include_top=False)
    model = VGG16(include_top=False, input_shape=input_shape)
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, activation='relu',
                   kernel_initializer='he_uniform')(flat1)
    output = Dense(10, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def train_model(train_data_dir, validation_data_dir, img_width, img_height, epochs, batch_size, model: Model, model_save_path):
    # reference https://www.geeksforgeeks.org/python-image-classification-using-keras/
    train_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=epochs)

    model.save(model_save_path)
    return history


def plot_history(history, model_save_path):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_save_path, "model_accuracy.png"))
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_save_path, "model_loss.png"))


img_width, img_height = 290, 325
classes = ["A1", "A3", "B1", "B3", "C1", "C3", "None"]


train_data_dir, validation_data_dir, model_save_path = prep_files_for_training(
    classes)

train_size = get_class_stats(classes, train_data_dir)[classes[0]]
validation_size = get_class_stats(classes, validation_data_dir)[classes[0]]

epochs = 200
batch_size = 16

model = compile_model(img_width, img_height)

history = train_model(train_data_dir, validation_data_dir, img_width,
                      img_height, epochs, batch_size, model, model_save_path)

plot_history(history, model_save_path)

shutil.rmtree(train_data_dir)
shutil.rmtree(validation_data_dir)
