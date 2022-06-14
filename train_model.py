import os

import plotly.express as px
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame

from class_statistics import get_class_stats, save_stats_to_file
from file_utils import clean_up_after_training, prep_files_for_training



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
    output = Dense(36, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_model(train_data_dir, validation_data_dir, img_width, img_height, epochs, batch_size, model: Model, model_save_path):
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


def save_model_history(history, model_save_path, train_class_stats, validation_class_stats):
    # save stats to file
    save_stats_to_file(
        train_class_stats,
        os.path.join(model_save_path, "train_class_stats.txt")
    )
    save_stats_to_file(
        validation_class_stats,
        os.path.join(model_save_path, "validation_class_stats.txt")
    )

    # save accuracy and loss charts to file
    accuracy_history = DataFrame({
        'epoch': history.epoch,
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })
    fig = px.line(
        accuracy_history,
        x="epoch",
        y=[accuracy_history['accuracy'], accuracy_history['val_accuracy']],
        title=f"Accuracy history - {os.path.basename(model_save_path)}"
    )
    fig.write_html(os.path.join(model_save_path, "model_accuracy.html"))

    loss_history = DataFrame({
        'epoch': history.epoch,
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    fig = px.line(
        loss_history,
        x="epoch",
        y=[loss_history['loss'], loss_history['val_loss']],
        title=f"Loss history - {os.path.basename(model_save_path)}"
    )
    fig.write_html(os.path.join(model_save_path, "model_loss.html"))


def main(img_width, img_height, classes, epochs, batch_size, normalize=False):
    # collect stats on and prepare training and validation files
    train_data_dir, validation_data_dir, model_save_path = prep_files_for_training(
        classes, normalize, base_path=os.path.dirname(os.path.realpath(__file__)))

    train_class_stats = get_class_stats(classes, train_data_dir)
    validation_class_stats = get_class_stats(classes, validation_data_dir)

    # create and train model, save training info
    model = compile_model(img_width, img_height)

    history = fit_model(train_data_dir, validation_data_dir, img_width,
                        img_height, epochs, batch_size, model, model_save_path)

    save_model_history(history, model_save_path,
                       train_class_stats, validation_class_stats)

    # clean up
    clean_up_after_training(train_data_dir, validation_data_dir)


if __name__ == "__main__":
    # image and class params
    img_width, img_height = 290, 325
    classes = ["A1", "A3", "B1", "B3", "C1", "C3", "None"]

    # training params
    epochs = 7
    batch_size = 24

    main(img_width, img_height, classes, epochs, batch_size)
