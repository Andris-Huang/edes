import importlib

def load_device_lib(path="/home/electron/artiq/experiment/devices/base.py"):
    spec = importlib.util.spec_from_file_location("device_lib", path)
    device_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device_lib)
    return device_lib