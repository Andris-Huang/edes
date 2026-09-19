import os
import importlib.util
import edes
import pandas as pd
import re
from datetime import datetime, time
from types import SimpleNamespace
import h5py
import fnmatch
import numpy as np
import json

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

def list_dirnames(dir_path):
    """Returns a list of subfolder names in the specified directory."""
    return [
        entry.name for entry in os.scandir(dir_path) 
        if entry.is_dir()
    ]

def get_artiq_event_files(base_dir, artiq_id):
    # Search target pattern, e.g., "artiq_11143"
    target_pattern = f"artiq_{artiq_id}"
    
    # 1. Get subfolder names and find the matching directory
    subfolders = list_dirnames(base_dir)
    matched_folder = None
    
    for folder in subfolders:
        if target_pattern in folder:
            matched_folder = os.path.join(base_dir, folder) if not folder.startswith(base_dir) else folder
            break
            
    if not matched_folder:
        print(f"No directory found matching ID: {artiq_id}")
        return []
    
    # 2. Get files inside matching folder and filter for "event_"
    all_files = list_filenames(matched_folder)
    event_files = [f'{matched_folder}/{f}' for f in all_files if "event_" in f and "mr" not in f]
    
    return event_files 

def list_filepaths(folder_path):
    """Return a list of full file paths in the specified folder."""
    try:
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def list_files_in_time_range(folder_path, name="*", start_time=None, stop_time=None):
    """
    Returns absolute paths of files in folder_path where the filename starts with 
    a matching prefix and the time in the filename (_hr_min_sec) falls between 
    start_time and stop_time.
    
    :param folder_path: Path to the target directory.
    :param name: Prefix string or wildcard pattern (e.g., 'report', 'log_*'). Defaults to '*' (all names).
    :param start_time: Start time as "HH:MM:SS" string, datetime.time, or None.
    :param stop_time: Stop time as "HH:MM:SS" string, datetime.time, or None.
    :return: A list of absolute filepaths.
    """
    # 1. Convert string times to datetime.time objects if provided
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    if isinstance(stop_time, str):
        stop_time = datetime.strptime(stop_time, "%H:%M:%S").time()

    # 2. Regex pattern to capture the _hr_min_sec pattern and isolate the prefix
    # Pattern explanation:
    # ^(.*?): Captures the prefix/name at the beginning of the string (Group 1)
    # _(\d{1,2})_(\d{1,2})_(\d{1,2}): Captures hours, minutes, and seconds (Groups 2, 3, 4)
    # (?:\.[^.]+)?$: Handles an optional file extension at the very end
    time_pattern = re.compile(r'^(.*?)_(\d{1,2})-(\d{1,2})-(\d{1,2})(?:\.[^.]+)?$')
    
    matching_filepaths = []
    # print(os.listdir(folder_path))
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        
        if os.path.isfile(full_path):
            match = time_pattern.search(f)
            # print(match)
            
            if match:
                # Extract the leading prefix and the time components
                file_prefix, hr, mnt, sec = match.groups()
                hr = int(hr) 
                mnt = int(mnt) 
                sec = int(sec)
                
                # 3. Check if the file's prefix matches the 'name' constraint
                # fnmatch handles '*' automatically
                if not fnmatch.fnmatch(file_prefix, name):
                    continue
                
                try:
                    # print(hr, mnt, sec)
                    file_time = time(hr, mnt, sec)
                    
                    # 4. Determine bounds conditionally based on None values
                    if start_time is not None and stop_time is not None:
                        if start_time <= stop_time:
                            in_range = start_time <= file_time <= stop_time
                        else:
                            # Over-the-midnight span (e.g., 22:00:00 to 02:00:00)
                            in_range = file_time >= start_time or file_time <= stop_time
                    elif start_time is not None:
                        in_range = file_time >= start_time
                    elif stop_time is not None:
                        in_range = file_time <= stop_time
                    else:
                        in_range = True
                    
                    if in_range:
                        # print(1)
                        matching_filepaths.append(os.path.abspath(full_path))
                        
                except ValueError:
                    # Skip if parsed numbers don't form a valid clock time
                    continue
                    
    return matching_filepaths
    

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
    
def load_h5_data(filepath, base=None):
    if base is not None:
        filepath = os.path.join(base, filepath)
        
    data = {}
    try:
        with h5py.File(filepath, 'r') as f:
            def _read_node(node):
                if isinstance(node, h5py.Group):
                    return {k: _read_node(v) for k, v in node.items()}
                elif isinstance(node, h5py.Dataset):
                    # Use np.array(node) instead of node[()] to safely load compound types
                    val = np.array(node)
                    # If it's a 0-D scalar array, unpack it
                    return val.item() if val.shape == () else val
                return None

            for key in f.keys():
                data[key] = _read_node(f[key])
                
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
    
def load_exp_config(config_file): 
    return load_lib(f"{edes.__config_folder__}/{config_file}.py")

def value_extraction(key): 
    pattern = r"dBm(?P<dBm_val>[-+]?\d*\.\d+|\d+)_average(?P<average>\d+)_repetition(?P<repetition>\d+)_data"
    match = re.match(pattern, key)
    data = match.groupdict()
    
    # Convert extracted strings to their proper numerical types
    dBm_Valon = float(data['dBm_val'])
    average = int(data['average'])
    repetition = int(data['repetition'])
    return dBm_Valon, average, repetition

def convert_dict_to_dataframe(input_dict):
    # Regex pattern to capture the values and determine if it's data or time
    pattern = r"dBm(?P<dBm>[-+]?\d*\.\d+|\d+)_average(?P<avg>\d+)_repetition(?P<rep>\d+)_(?P<type>data|time)"
    
    # Dictionary to temporarily store grouped records
    grouped_records = {}
    
    for key, val_array in input_dict.items():
        match = re.match(pattern, key)
        if match:
            # print('here')
            metadata = match.groupdict()
            # Convert extracted strings to their proper types
            dBm_val = float(metadata['dBm'])
            avg_val = int(metadata['avg'])
            rep_val = int(metadata['rep'])
            data_type = metadata['type']  # This will be either 'data' or 'time'
            
            # Unique identifier for this specific combination
            combo_key = (dBm_val, avg_val, rep_val)
            
            # Initialize the row structure if we haven't seen this combination yet
            if combo_key not in grouped_records:
                grouped_records[combo_key] = {
                    'dBm': dBm_val,
                    'avg': avg_val,
                    'rep': rep_val,
                    'data': None,
                    'time': None
                }
            
            # Assign the numpy array to the correct column ('data' or 'time')
            grouped_records[combo_key][data_type] = val_array

    # Convert the dictionary values into a list of rows and create the DataFrame
    df = pd.DataFrame(list(grouped_records.values()))
    
    # Optional: Reorder columns explicitly to match your request
    # df = df[['dBm', 'avg', 'rep', 'data', 'time']]
    
    return df


def parse_encoded_number(val):
    """
    Cleans sweep string numbers like '_-0p4' or '_-0p09999999999999998'
    into clean floats like -0.4 and -0.1. Non-numeric strings are returned
    unchanged; float/int values are rounded the same way for consistency.
    """
    if not isinstance(val, str):
        if isinstance(val, (float, np.floating)):
            return round(float(val), 6)
        return val

    s = val.strip().strip('_')
    candidate = s.replace('p', '.').replace('P', '.')

    try:
        num = float(candidate)
        num = round(num, 6)
        return 0.0 if num == 0.0 else num
    except ValueError:
        return val


def convert_sweep_dict_to_dataframe(data_dict, var_names=None):
    """
    Converts an HDF5 data dictionary from a parameter sweep (e.g. DC-multipole
    scans) into a structured pandas DataFrame. Handles keys formatted like
    'data.Ex=10.Ey=-5', 'data.Ex_-0p1.Ey_-0p4' (encoded via parse_encoded_number),
    or plain positional tokens 'data.10.-5'.

    Not to be confused with :func:`convert_dict_to_dataframe` above, which
    parses a different, unrelated key schema
    ('dBm{val}_average{val}_repetition{val}_data|time') used by the lock-in
    detection pipeline.

    Parameters
    ----------
    data_dict : dict
        Dictionary loaded from HDF5 file.
    var_names : list of str, optional
        Custom column names for positional variables.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with dynamic variable columns, 'data', and 'time' (if present).
    """
    if var_names is None:
        for meta_key in ['var_names', 'variables', 'param_names']:
            if meta_key in data_dict:
                var_names = list(data_dict[meta_key])
                break

    time_data = None
    for t_key in ['time', 't', 't_axis', 'x_axis']:
        if t_key in data_dict:
            time_data = data_dict[t_key]
            break

    records = []
    datasets_source = data_dict.get('datasets', data_dict)

    for key, value in datasets_source.items():
        if not isinstance(key, str):
            continue

        if key.startswith('data.') or key.startswith('data/'):
            tokens = key.replace('/', '.').split('.')[1:]
            row = {}

            for idx, token in enumerate(tokens):
                if '=' in token:
                    k, v = token.split('=', 1)
                    row[k] = parse_encoded_number(v)
                elif '_' in token:
                    parts = token.split('_', 1)
                    parsed_val = parse_encoded_number(parts[1])
                    if isinstance(parsed_val, (int, float)):
                        row[parts[0]] = parsed_val
                    else:
                        col_name = var_names[idx] if (var_names and idx < len(var_names)) else f'var{idx + 1}'
                        row[col_name] = parse_encoded_number(token)
                else:
                    col_name = var_names[idx] if (var_names and idx < len(var_names)) else f'var{idx + 1}'
                    row[col_name] = parse_encoded_number(token)

            row['data'] = value
            if time_data is not None:
                row['time'] = time_data

            records.append(row)

    df = pd.DataFrame(records)

    for col in df.columns:
        if col not in ['data', 'time']:
            try:
                df[col] = pd.to_numeric(df[col].apply(parse_encoded_number))
            except (ValueError, TypeError):
                pass

    return df


def find_file_by_rid(rid, search_dir, base_dir_to_strip):
    """Searches for an ARTIQ .h5 file matching the given RID."""
    target_prefix = f"{int(rid):09d}-"
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.startswith(target_prefix) and file.endswith('.h5'):
                full_path = os.path.join(root, file)
                # Strip the home_dir from the start so load_h5_data works correctly with base=home_dir
                if full_path.startswith(base_dir_to_strip):
                    return full_path[len(base_dir_to_strip):]
                return full_path
    return None


def load_artiq_h5_arguments(filename, arg, base=''):
    return json.loads(load_h5_data(filename, base=base)['expid'])['arguments'][arg]