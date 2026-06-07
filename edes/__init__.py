import os
from edes.utils.file_handling import load_saving_dir

__base_dir__ = f'{os.path.dirname(__file__)}/..'
__config_folder__ = os.path.join(os.path.dirname(__file__), "experiments/configs")
__default_saving_dir__ = load_saving_dir()
__log_folder__ = os.path.join(__base_dir__, "logs")