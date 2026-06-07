import os
import importlib.util
import edes
from types import SimpleNamespace
import h5py

def dict_to_obj(data_dict):
    return SimpleNamespace(**data_dict)

def list_filenames(folder_path):
    """Return a list of filenames in the specified folder."""
    try:
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def load_lib(path=""):
    spec = importlib.util.spec_from_file_location("lib", path)
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib

def load_saving_dir():
    log_path = os.path.join(edes.__base_dir__, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    saving_dir = os.path.join(log_path, "saving_dir.txt")
    if os.path.exists(saving_dir):
        with open(saving_dir, "r") as f:
            return f.read().strip()
    else:
        with open(saving_dir, "w") as f:
            f.write(log_path)
        return log_path
    
def load_h5_data(filepath):
    """Load data from an HDF5 file and return it as a dictionary."""
    data = {}
    try:
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][()]
    except FileNotFoundError:
        print(f">>> File not found: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
    return data

def update_latest_filename(filename): 
    log_path = edes.__log_folder__
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    latest_file_path = os.path.join(log_path, "latest_file.txt")
    with open(latest_file_path, "w") as f:
        f.write(filename)

def load_latest_filename():
    log_path = edes.__log_folder__
    latest_file_path = os.path.join(log_path, "latest_file.txt")
    if os.path.exists(latest_file_path):
        with open(latest_file_path, "r") as f:
            return f.read().strip()
    else:
        print(f">>> No latest file found at {latest_file_path}")
        return None

def load_latest_data():
    latest_file = load_latest_filename()
    if latest_file is not None:
        data = load_h5_data(latest_file)
        data['filename'] = latest_file 
        return data
    else:
        return None