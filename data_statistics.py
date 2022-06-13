import os

module_path = os.path.dirname(os.path.realpath(__file__))
train_data_dir = os.path.join(module_path, 'data', 'train')
validation_data_dir = os.path.join(module_path, 'data', 'validate')


def get_class_stats(classes, data_dir) -> dict[str, int]:
    stats = dict()
    for class_name in classes:
        class_data_dir = os.path.join(data_dir, class_name)

        file_names = [os.path.join(class_data_dir, file_name) for file_name in next(
            os.walk(class_data_dir), (None, None, []))[2]]
        stats[class_name] = len(file_names)
    return stats


def print_statistics(stats):
    for key in stats:
        print(f"{key} : {stats[key]}")


if __name__ == "__main__":
    classes = ["A1", "A3", "B1", "B3", "C1", "C3", "None"]
    class_stats = get_class_stats(classes, train_data_dir)
    print_statistics(class_stats)
