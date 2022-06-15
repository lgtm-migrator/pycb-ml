import os

import plotly.express as px
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame

from pycbml.class_statistics import class_stats, print_statistics


def create_VGG16_transfer_model(classes, input_shape):
    model = VGG16(weights='imagenet', include_top=False,
                  input_shape=input_shape
                  )

    x = Flatten()(model.layers[-1].output)
    x = Dense(128, activation='relu',
              kernel_initializer='he_uniform')(x)
    x = Dense(64, activation='relu',
              kernel_initializer='he_uniform')(x)
    x = Dense(len(classes), activation='sigmoid')(x)

    model = Model(model.inputs, x)
    return model


def create_simple_model(classes, input_shape):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dense(len(classes), activation='sigmoid')
    ])


def compile_model(img_width, img_height, classes):
    print("Compiling model...")

    input_shape = (img_width, img_height, 3)

    model = create_VGG16_transfer_model(classes, input_shape)
    #model = create_simple_model(classes, input_shape)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def fit_model(train_data_dir, img_width, img_height, epochs, batch_size, model: Model, model_save_path):
    # reference https://www.geeksforgeeks.org/python-image-classification-using-keras/
    # https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_directory/

    train_datagen = ImageDataGenerator(validation_split=0.20)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='rgb',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='rgb',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,
        subset='validation'
    )

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


def save_model_history(history, model_save_path):
    print("Saving training history plots...")

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
        title=f"Accuracy history - {os.path.basename(model_save_path)}",
        template="plotly_dark"
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
        title=f"Loss history - {os.path.basename(model_save_path)}",
        template="plotly_dark"
    )
    fig.write_html(os.path.join(model_save_path, "model_loss.html"))
    return os.path.basename(model_save_path)


def prepare_files(base_path):

    train_data_dir = os.path.join(os.getcwd(), "train")
    model_save_path = os.path.join(base_path, "models")
    return train_data_dir, model_save_path


def main(img_width, img_height, classes, epochs, batch_size, normalize=False, base_path=os.path.dirname(os.path.realpath(__file__))):
    train_data_dir, model_save_path = prepare_files(base_path)
    print_statistics(class_stats(classes, train_data_dir))

    # create and train model, save training info
    model = compile_model(img_width, img_height, classes)

    history = fit_model(train_data_dir, img_width, img_height,
                        epochs, batch_size, model, model_save_path)

    return save_model_history(history, model_save_path)


if __name__ == "__main__":
    # image and class params
    img_width, img_height = 290, 325
    classes = ["A1", "A3", "B1", "B3", "C1", "C3", "None"]

    # define working directory
    base_path = "/content/drive/MyDrive/Programming/Python/pycb-ml/"

    # training params
    epochs = 5
    batch_size = 32
    main(img_width,
         img_height,
         classes,
         epochs,
         batch_size,
         base_path=base_path)
