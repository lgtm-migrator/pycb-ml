import os
import zipfile
from tqdm import tqdm


def unzip_data(base_path, overwrite):
    zip_path = os.path.join(base_path, "data.zip")
    destination = os.getcwd()
    data_dir = os.path.join(destination, "data")
    if not os.path.exists(data_dir) or overwrite:
        with zipfile.ZipFile(zip_path) as zf:
            for member in tqdm(zf.infolist(), desc=f'Extracting {zip_path}'):
                try:
                    zf.extract(member, destination)
                except zipfile.error as e:
                    pass
    else:
        print("Data directory already exits. Specify overwrite to overwrite.")
    return data_dir


def prepare_files(base_path, overwrite=False):
    data_dir = unzip_data(base_path, overwrite)
    model_save_path = os.path.join(base_path, "models")
    return data_dir, model_save_path
