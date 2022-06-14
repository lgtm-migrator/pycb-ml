import os
import random
import shutil
import tempfile
import time

from class_statistics import get_class_stats, print_statistics


def copy_list_of_files(file_list, desitation_dir):
    for file_name in file_list:
        shutil.copy2(file_name, desitation_dir)


def copy_files_to_temp(classes, train_data_dir, normalize=False):
    temp_train_dir = tempfile.mkdtemp()
    temp_validate_dir = tempfile.mkdtemp()

    class_stats = get_class_stats(classes, train_data_dir)

    # get value to normalize training data count
    smallest_class_size = min(class_stats.values())

    for class_name in classes:
        class_data_dir = os.path.join(train_data_dir, class_name)

        # make dir for class in temp dirs
        os.makedirs(os.path.join(temp_train_dir, class_name))
        os.makedirs(os.path.join(temp_validate_dir, class_name))

        # get file list for class cut to smallest class size
        file_names = [os.path.join(class_data_dir, file_name) for file_name in next(
            os.walk(class_data_dir), (None, None, []))[2]]
        if normalize:
            file_names = file_names[:smallest_class_size]

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

    print("Training class stats:")
    print_statistics(get_class_stats(classes, temp_train_dir))
    print("Validation class stats:")
    print_statistics(get_class_stats(classes, temp_validate_dir))

    return temp_train_dir, temp_validate_dir


def prep_files_for_training(classes, normalize=False, base_path = os.path.dirname(os.path.realpath(__file__))):
    train_data_dir = os.path.join(base_path, 'data', 'train')

    model_save_path = os.path.join(
        base_path,
        "models",
        f"save{str(int(time.time()))}"
    )

    temp_train_dir, temp_validate_dir = copy_files_to_temp(
        classes, train_data_dir, normalize)

    return temp_train_dir, temp_validate_dir, model_save_path


def clean_up_after_training(train_data_dir, validation_data_dir):
    shutil.rmtree(train_data_dir)
    shutil.rmtree(validation_data_dir)