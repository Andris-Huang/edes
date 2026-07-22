from edes.utils.file_handling import load_lib, dict_to_obj
from edes.experiments.devices import base
import edes
import datetime
import os



class Experiment:
    """
    Experiment object to store and load all relevant devices.
    
    Functions
    ---
    * load_devices()
        - Load all devices specified in the given config file
    * load_device(device_name)
        - Load specific device previously loaded as <device_name>
    * list_devices()
        - List all connected devices
    * close_all()
        - Close all connections
    """
    def __init__(self, config_file, saving_dir=None, log_callback=None):
        self.log_callback = log_callback if log_callback else print
        self.config = load_lib(f"{edes.__config_folder__}/{config_file}.py")
        if saving_dir is not None:
            self.saving_dir = saving_dir
        elif hasattr(self.config, "saving_dir"):
            self.saving_dir = self.config.saving_dir
        else:
            self.saving_dir = edes.__default_saving_dir__
        os.makedirs(self.saving_dir, exist_ok=True)
        self.devices = self.load_devices()
    
    def load_devices(self): 
        devices = {}
        for device_name in [i for i in dir(self.config) if not i.startswith('_') and i is not None]:
            item = getattr(self.config, device_name)
            if type(item) is dict and "device" in item and "addr" in item:
                device_class = getattr(base, getattr(self.config, device_name)['device'])
                try:
                    devices[device_name] = device_class(getattr(self.config, device_name)['addr'], 
                                                **getattr(self.config, device_name).get('params', {}), log_callback=self.log_callback)
                except Exception as e:
                    self.log_callback(f">>> ERROR initializing device {device_name}: {e}")
        for device_name in devices:
            setattr(self, device_name, devices[device_name])
        return devices
    
    def load_device(self, device_name):
        if hasattr(self, device_name):
            getattr(self, device_name).close()
        item = getattr(self.config, device_name)
        if type(item) is dict and "device" in item and "addr" in item:
            device_class = getattr(base, getattr(self.config, device_name)['device'])
            try:
                self.devices[device_name] = device_class(getattr(self.config, device_name)['addr'], 
                              **getattr(self.config, device_name).get('params', {}))
                setattr(self, device_name, self.devices[device_name])
            except Exception as e:
                self.log_callback(f">>> ERROR initializing device {device_name}: {e}")

    def list_devices(self):
        return list(self.devices.keys())

    def close_all(self):
        for device in self.devices.values():
            try:
                device.close()
            except Exception as e:
                self.log_callback(f">>> ERROR closing device: {e}")