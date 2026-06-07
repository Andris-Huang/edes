from unittest import result

import numpy as np
import time
from tqdm import tqdm, trange
import edes
from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import json
import h5py

class Sequence:
    def __init__(self, name, saving_dir=None, **kwargs):
        self.name = name
        self.saving_dir = saving_dir if saving_dir is not None else edes.utils.file_handling.load_saving_dir()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        return f"Sequence(name={self.name})"

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # def run_save(self): 
    #     result = self.run()
    #     np.savez(f"{self.saving_dir}/{self.name}_{time.strftime('%H-%M-%S')}.npz", **result)
    #     return result
    def run_save(self):
        result = self.run()
        timestamp = time.strftime('%H-%M-%S')
        filepath = f"{self.saving_dir}/{self.name}_{timestamp}.h5"
        edes.utils.file_handling.update_latest_filename(filepath)
        
        with h5py.File(filepath, 'w') as f:
            # save data
            for key, val in result.items():
                f.create_dataset(key, data=val)

            for key, val in self.parameters.items():
                f.create_dataset(key, data=val)
            
            # # config as attributes
            # if hasattr(self, 'config') and self.config is not None:
            #     if is_dataclass(self.config):
            #         for key, val in asdict(self.config).items():
            #             f.attrs[key] = val
        
        return result

# @dataclass
# class TipVoltageSweepConfig:
#     V_start: float = 0
#     V_stop: float = 100
#     V_step: float = 10
#     R: float = 100e6
#     N_avg: int = 8
#     t_PSU_settle: float = 2
#     t_meas_delay: float = 0.2

class TipVoltageSweep(Sequence):
    def __init__(self, name='TipVoltageSweep', V_start=0, 
                 V_stop=100, V_step=10, R=100e6, N_avg=8, 
                 t_PSU_settle=2, t_meas_delay=0.2,
                 FEtip_PSU=None, multimeter=None, **kwargs):
        super().__init__(name, **kwargs)
        self.V_start = V_start
        self.V_stop = V_stop
        self.V_step = V_step
        self.N_avg = N_avg 
        self.t_PSU_settle = t_PSU_settle
        self.t_meas_delay = t_meas_delay
        self.FEtip_PSU = FEtip_PSU
        self.multimeter = multimeter
        self.R = R
        self.parameters = {
            'V_start': V_start,
            'V_stop': V_stop,
            'V_step': V_step,
            'N_avg': N_avg,
            'R': R, 
            't_PSU_settle': t_PSU_settle,
            't_meas_delay': t_meas_delay, 
            'FEtip_PSU': FEtip_PSU.name,
            'multimeter': multimeter.name
        }
    # def __init__(self, config=None, FEtip_PSU=None, multimeter=None, **kwargs):
    #     if config is None:
    #         config = TipVoltageSweepConfig(
    #             V_start=kwargs.pop('V_start', 0),
    #             V_stop=kwargs.pop('V_stop', 100),
    #             V_step=kwargs.pop('V_step', 10),
    #             R=kwargs.pop('R', 100e6),
    #             N_avg=kwargs.pop('N_avg', 8),
    #             t_PSU_settle=kwargs.pop('t_PSU_settle', 2),
    #             t_meas_delay=kwargs.pop('t_meas_delay', 0.2),
    #         )
    #     super().__init__(name='TipVoltageSweep', **kwargs)
    #     self.config = config
    #     self.FEtip_PSU = FEtip_PSU
    #     self.multimeter = multimeter

    def run(self):
        all_I = [] 
        all_V = np.arange(self.V_start, self.V_stop, self.V_step)
        for V in tqdm(all_V): 
            self.FEtip_PSU.set_voltage(V) 
            time.sleep(self.t_PSU_settle) 
            I_loc = []
            for _ in range(self.N_avg):
                I = self.multimeter.measure_V()/self.R
                I_loc.append(I)    
                time.sleep(self.t_meas_delay) 
            all_I.append(I_loc) 
        self.FEtip_PSU.ramp_down(0)
        return {'all_V': all_V, 'all_I': np.array(all_I)}