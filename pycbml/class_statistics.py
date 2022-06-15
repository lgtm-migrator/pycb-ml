import os
from tabulate import tabulate

base_path = os.path.dirname(os.path.realpath(__file__))
train_data_dir = os.path.join(base_path, 'data', 'train')
validation_data_dir = os.path.join(base_path, 'data', 'validate')


def class_stats(classes, data_dir):
    stats = dict()
    for class_name in classes:
        class_data_dir = os.path.join(data_dir, class_name)

        file_names = [os.path.join(class_data_dir, file_name) for file_name in next(
            os.walk(class_data_dir), (None, None, []))[2]]
        stats[class_name] = len(file_names)
    return stats


def print_statistics(stats):
    data = [[[key, stats[key]] for key in stats]]
    print(tabulate(data, headers=["Class Name", "Number of Images"]))


if __name__ == "__main__":
    classes = ["A1", "A3", "B1", "B3", "C1", "C3", "None"]
    print_statistics(class_stats(classes, train_data_dir))
