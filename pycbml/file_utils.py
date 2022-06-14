import os
import random
import shutil
import tempfile
import time

from tqdm import tqdm

from pycbml.class_statistics import class_stats, print_statistics


def copy_list_of_files_to_temp(file_list, temp_dir):
    for file_name in tqdm(file_list):
        class_dir = os.path.basename(os.path.dirname(file_name))
        shutil.copy(
            file_name,
            os.path.join(temp_dir, class_dir)
        )


def copy_files_to_temp(classes, train_sample, validation_sample):
    temp_train_dir = tempfile.mkdtemp()
    temp_validate_dir = tempfile.mkdtemp()
    for class_name in classes:

        # make dir for class in temp dirs
        os.makedirs(os.path.join(temp_train_dir, class_name))
        os.makedirs(os.path.join(temp_validate_dir, class_name))

    # copy samples into directories
    print("Copying files to temporary training dir...")
    copy_list_of_files_to_temp(train_sample, temp_train_dir)
    print("Copying files to temporary validation dir...")
    copy_list_of_files_to_temp(validation_sample, temp_validate_dir)
    return temp_train_dir, temp_validate_dir


def split_training_files(classes, train_data_dir, normalize=False,  validation_rate=0.25):

    # get value to normalize training data count
    smallest_class_size = min(class_stats(
        classes, train_data_dir).values()) if normalize else None

    validation_sample: list[str] = []
    train_sample: list[str] = []

    for class_name in classes:
        class_data_dir = os.path.join(train_data_dir, class_name)

        # get file list for class cut to smallest class size
        file_names = [os.path.join(class_data_dir, file_name) for file_name in next(
            os.walk(class_data_dir), (None, None, []))[2]]
        if normalize:
            file_names = file_names[:smallest_class_size]

        # calculate sample sizes
        validate_sample_size = int(
            len(file_names) * validation_rate)

        # create train and validate samples
        class_validate_sample = random.sample(
            file_names, validate_sample_size)
        class_train_sample = [
            x for x in file_names if x not in class_validate_sample]

        # add samples to list
        train_sample += class_train_sample
        validation_sample += class_validate_sample
    return train_sample, validation_sample


def prep_files_for_training(classes, normalize=False, base_path=os.path.dirname(os.path.realpath(__file__)), use_temp_dir=False):
    print("Preparing files for training...")
    train_data_dir = os.path.join(base_path, 'data', 'train')

    model_save_path = os.path.join(
        base_path,
        "models",
        f"save{str(int(time.time()))}"
    )

    train_sample, validation_sample = split_training_files(
        classes,
        train_data_dir,
        normalize,
        validation_rate=0.25
    )

    train_dir, validate_dir = copy_files_to_temp(
        classes, train_sample, validation_sample)

    print_statistics(class_stats(classes, train_dir))
    print_statistics(class_stats(classes, validate_dir))

    return train_dir, validate_dir, model_save_path


def clean_up_after_training(train_data_dir, validation_data_dir):
    shutil.rmtree(train_data_dir)
    shutil.rmtree(validation_data_dir)
