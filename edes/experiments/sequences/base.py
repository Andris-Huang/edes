from unittest import result

import numpy as np
import time
from tqdm import tqdm, trange
import edes
from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import json
import h5py
from edes.experiments.devices.base import Instrument
from edes.utils.circuits import dBm_to_Vp, dBm_to_W, Vp_to_dBm
from typing import TYPE_CHECKING
import time
from time import sleep
from edes.modules.variable_attenuator import var_attn_utils
from edes.utils import utils

# 1. This code block ONLY runs in your IDE for auto-complete
# It is completely ignored when the code actually runs
if TYPE_CHECKING:
    from edes.experiments.devices.base import *

class Sequence:
    def __init__(self, name, saving_dir=None, log_callback=None):
        self.name = name
        self.saving_dir = saving_dir if saving_dir is not None else edes.utils.file_handling.load_saving_dir()
        self.log_callback = log_callback
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
    
    def __str__(self):
        return f"Sequence(name={self.name})"

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def progress_bar(self, *args, **kwargs): 
        return tqdm(*args, **kwargs, file=self.log_callback)    

    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_parameters(self): 
        self.parameters = {k: v for k, v in vars(self).items() if k != 'parameters'}
        for k, v in self.parameters.items(): 
            if not isinstance(v, (int, float, str, np.ndarray)):
                if isinstance(v, dict): 
                    for ki, vi in v: 
                        self.parameters[ki] = vi 
                elif isinstance(v, type) and issubclass(v, Sequence): 
                    sub_params = v.get_parameters() 
                    for ki, vi in sub_params: 
                        if ki not in self.parameters:
                            self.parameters[ki] = vi
                else:
                    self.parameters[k] = str(v)
        return self.parameters
    
    def run_save(self):
        result = self.run()
        timestamp = time.strftime('%H-%M-%S')
        filepath = f"{self.saving_dir}/{self.name}_{timestamp}.h5"
        edes.utils.file_handling.update_latest_filename(filepath)
        self.parameters = self.get_parameters()
        
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
    
class GUITest(Sequence):
    def __init__(self, name='GUITest', i_start=0, i_end=1, **kwargs): 
        super().__init__(name, **kwargs)
        self.i_start = i_start 
        self.i_end = i_end
    
    def run(self): 
        self.log_callback('Has logger')
        for _ in self.progress_bar(range(self.i_start, self.i_end)): 
            time.sleep(1e-3)
        return {}


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
        

    def run(self):
        all_I = [] 
        all_V = np.arange(self.V_start, self.V_stop, self.V_step)
        for V in self.progress_bar(all_V): 
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
    

class TipVoltageSweepDifferential(TipVoltageSweep):
    def __init__(self, name='TipVoltageSweepDifferential', V_start=0, 
                 V_stop=100, V_step=10, R=100e6, N_avg=8, I_max=10e-9,
                 t_PSU_settle=2, t_meas_delay=0.2,
                 ch_sweep='neg', ch_fixed='pos', V_fixed=0,
                 FEtip_PSU=None, multimeter=None, **kwargs):
        super().__init__(name, V_start, V_stop, V_step, R, N_avg, t_PSU_settle, t_meas_delay, FEtip_PSU, multimeter, **kwargs)
        self.ch_sweep = ch_sweep
        self.ch_fixed = ch_fixed
        self.V_fixed = V_fixed
        self.FEtip_PSU.ramp_up_ch(self.ch_fixed, self.V_fixed)
        self.FEtip_PSU.select_ch(self.ch_sweep)
        self.I_max = I_max
        

    def run(self):
        self.FEtip_PSU.ramp_up_ch(self.ch_fixed, self.V_fixed)
        self.FEtip_PSU.select_ch(self.ch_sweep)
        all_I = [] 
        # all_V = np.arange(self.V_start, self.V_stop, self.V_step)
        # 1. Figure out exactly how many clean steps fit between start and stop
        num_steps = int(np.around((self.V_stop - self.V_start) / self.V_step)) + 1

        # 2. Create an array of clean integers [0, 1, 2, ..., num_steps - 1]
        # 3. Multiply by the step and add the start value
        all_V = self.V_start + np.arange(num_steps) * self.V_step
        for V in self.progress_bar(all_V): 
            self.FEtip_PSU.set_voltage(V) 
            time.sleep(self.t_PSU_settle) 
            I_loc = []
            for _ in range(self.N_avg):
                I = self.multimeter.measure_V()/self.R
                I_loc.append(I)    
                time.sleep(self.t_meas_delay) 
            all_I.append(I_loc) 
            if I >= self.I_max: 
                print(f">>> Warning: Current {I} exceeds maximum allowed {self.I_max}. Stopping sweep.")
                break
        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch(self.ch_fixed, 0)
        return {'all_V': all_V, 'all_I': np.array(all_I)}
    

class RFPowerSweep(Sequence): 
    """
    Sweep the RF output power based on specificed power levels.
    """
    def __init__(self, name='RFPowerSweep', saving_dir=None, 
                 t_power_switch=0.1, P_min_dBm=1, P_max_dBm=10, 
                 steps=800, Valon=None, **kwargs):
        """
        Parameters
        --- 
        * t_power_switch : [s]
            The time gap between two power levels
        * P_min_dBm : [dBm] 
            The min values for power to sweep in dBm
        * P_max_dBm : [dBm]
            The upper bound for sweeping in dBm
        * steps : [int] 
            The total number of steps to sweep
        """
        super().__init__(name, saving_dir=saving_dir, **kwargs)
        self.t_power_switch = t_power_switch
        self.P_min_dBm = P_min_dBm
        self.P_max_dBm = P_max_dBm 
        self.steps = steps 
        self.Valon = Valon
    
    def run(self, dBm_scan=None): 
        if dBm_scan is None: 
            dBm_scan = np.linspace(self.P_min_dBm, self.P_max_dBm, self.steps)
        for P_dbm in dBm_scan:
            self.Valon.set_power(P_dbm)
            time.sleep(self.t_power_switch)
        return {"dBm_scan": dBm_scan}
    

class RFPowerSweepVolLinear(Sequence): 
    def __init__(self, name='RFPowerSweep', saving_dir=None,
                 t_valon_settle=8, t_meas_delay=8, t_power_switch=0.1, 
                 t_data_buffer=1,
                 P_min_dBm = 1, P_max_dBm = 10, R=50, steps=800, N_avg=5, 
                 V_on=1400, V_off=700,
                 SSA_freq_center = None, SSA_freq_span = None,
                 SSA_RBW = None, SSA_SWT=None,
                 Valon: 'Valon | None' =None, SSA: 'SSA3032X_R | None'=None, FEtip_PSU: 'PS350_viaDP832A | None'=None,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, **kwargs)
        self.t_valon_settle = t_valon_settle
        self.t_meas_delay = t_meas_delay
        self.t_power_switch = t_power_switch
        self.t_data_buffer = t_data_buffer
        self.P_min_dBm = P_min_dBm
        self.P_max_dBm = P_max_dBm
        self.V_min = dBm_to_Vp(self.P_min_dBm)
        self.V_max = dBm_to_Vp(self.P_max_dBm)
        self.R = R
        self.steps = steps
        self.N_avg = N_avg 
        self.V_on = V_on
        self.V_off = V_off
        self.SSA_freq_center = SSA_freq_center if SSA_freq_center is not None else SSA.default_freq_center
        self.SSA_freq_span = SSA_freq_span if SSA_freq_span is not None else SSA.default_freq_span 
        self.SSA_RBW = SSA_RBW if SSA_RBW is not None else SSA.default_RBW
        self.SSA_SWT = SSA_SWT if SSA_SWT is not None else SSA.default_SWT
        self.Valon = Valon
        self.SSA = SSA
        self.FEtip_PSU = FEtip_PSU
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        
    # def test(self):
    #     print((self.V_min,self.V_max))

    def SSA_init(self, freq_center, span, RBW, SWT):
        self.SSA.set_freq_center(freq_center)
        self.SSA.set_freq_span(span)
        self.SSA.set_RBW(RBW)
        self.SSA.set_SWT(SWT)
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        voltage_scan = np.linspace(self.V_min, self.V_max, self.steps)
        dBm_scan = Vp_to_dBm(voltage_scan,R=self.R)
        self.FEtip_PSU.set_voltage(self.V_on)
        all_meas = []
        all_t_start = [] 
        all_t_end = []
        self.Valon.set_power(dBm_scan[0])
        self.Valon.output_on()
        for _ in self.progress_bar(range(self.N_avg)):
            self.Valon.set_power(dBm_scan[0])
            time.sleep(self.t_valon_settle)
            self.SSA.clear_averaging()
            self.SSA.get_trace()
            t0 = time.time()
            for P_dbm in dBm_scan:
                self.Valon.set_power(P_dbm)
                time.sleep(self.t_power_switch)
            t1 = time.time()
            self.Valon.set_power(dBm_scan[0])
            time.sleep(self.t_meas_delay)
            data = self.SSA.get_trace()
            all_meas.append(data)
            all_t_start.append(t0)
            all_t_end.append(t1)
        self.FEtip_PSU.ramp_down(0)
        self.Valon.set_power(0)
        self.Valon.output_off()

        return {'dBm_scan': dBm_scan, 'all_meas': np.array(all_meas), 't_start': np.array(all_t_start), 't_end': np.array(all_t_end)}
    

class RFPowerSweepVolLinTipSwitched(RFPowerSweepVolLinear): 
    def __init__(self, name='RFPowerSweep',
                 **kwargs): 
        super().__init__(name, **kwargs)

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        voltage_scan = np.linspace(self.V_min, self.V_max, self.steps)
        dBm_scan = Vp_to_dBm(voltage_scan,R=self.R)
        all_meas = []
        all_t_start = [] 
        all_t_end = []
        self.Valon.set_power(dBm_scan[0])
        self.Valon.output_on()
        for _ in self.progress_bar(range(self.N_avg)):
            self.Valon.set_power(dBm_scan[0])
            self.Valon.output_on()
            self.FEtip_PSU.set_voltage(self.V_on)
            self.SSA.clear_averaging()
            self.SSA.get_trace()
            time.sleep(self.t_valon_settle)
            t0 = time.time()
            for P_dbm in dBm_scan:
                self.Valon.set_power(P_dbm)
                time.sleep(self.t_power_switch)
            t1 = time.time()
            self.Valon.set_power(dBm_scan[0])
            self.FEtip_PSU.set_voltage(self.V_off) 
            time.sleep(self.t_meas_delay)
            data = self.SSA.get_trace()
            self.Valon.output_off()
            all_meas.append(data)
            all_t_start.append(t0)
            all_t_end.append(t1)
            time.sleep(self.t_valon_settle)
        self.FEtip_PSU.ramp_down(0)
        self.Valon.set_power(0)
        self.Valon.output_off()

        return {'dBm_scan': dBm_scan, 'all_meas': np.array(all_meas), 't_start': np.array(all_t_start), 't_end': np.array(all_t_end)}
    

class RigolAWGSweep(Sequence): 
    """
    A simple test sequence to verify the Rigol DG4062 hardware ramp 
    and hardware trigger output on an oscilloscope.
    """
    def __init__(self, name='RigolSweepTest', 
                 v_min=0.0, v_max=2.0, v_hold=0.0, v_flat=None,
                 flat_time=0.05, ramp_time=0.1, hold_time=0.05,
                 AWG=None, channel=1, running_time=None, 
                 **kwargs):
        """
        Parameters
        --- 
        * v_min : [V] -> Lower limit of the ramp
        * v_max : [V] -> Upper limit of the ramp
        * sweep_time : [s] -> Duration of one ramp-up cycle
        * AWG : Instrument -> The RigolDG4062 instance
        * channel : [int] -> The target AWG channel
        """
        super().__init__(name, saving_dir=None, **kwargs)
        self.v_min = v_min
        self.v_max = v_max
        self.ramp_time = ramp_time
        self.v_hold = v_hold 
        self.flat_time = flat_time 
        self.hold_time = hold_time
        self.AWG = AWG
        self.channel = channel
        self.running_time = running_time if running_time is not None else ramp_time + flat_time + hold_time
        self.AWG.configure_burst_custom_arb(channel=channel, 
                                            start_v=v_min, stop_v=v_max, hold_v=v_hold, flat_v=v_flat,
                                            flat_time=flat_time, ramp_time=ramp_time, hold_time=hold_time)
        self.AWG.output_on() 

    def run_continuous(self): 
        print(f"[{self.name}] Configuring Rigol DG4062...")
        
        # 1. Use the API to build out the hardware ramp configuration
        self.AWG.configure_ramp_sweep(
            channel=self.channel, 
            start_v=self.v_min, 
            stop_v=self.v_max, 
            sweep_time=self.ramp_time
        )
        
        # 2. Fire the hardware output
        print(f"[{self.name}] Turning Channel {self.channel} ON.")
        
        
        print(f"[{self.name}] Running continuous hardware sweep test for 10 seconds.")
        print("Check your oscilloscope now!")
        self.AWG.set_output(channel=self.channel, state=True)
        # Let it loop continuously on the hardware level so you can inspect the scope
        time.sleep(self.running_time)
        
        # 3. Clean up and turn off after the test duration
        self.AWG.set_output(channel=self.channel, state=False)
        print(f"[{self.name}] Test complete. Output turned OFF.")
        
        return {"status": "Test Completed Successfully"}
    
    def run(self): 
        """
        Run a single voltage ramp
        """
        self.AWG.fire_burst()
        return {}
    

class RFPowerSweepAWG(RFPowerSweepVolLinTipSwitched): 
    def __init__(self, name='RFPowerSweepAWG', saving_dir=None,
                 P_min_dBm=-10, P_max_dBm =0, P_hold_dBm=-9,
                 ramp_time=0.1, hold_time=0.05,
                 AWG=None, P_valon=5.0,
                 t_data_buffer=1, t_seq_gap=0.5, N_avg=5, 
                 V_on=1400, V_off=700, t_tip_settle=3,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW = 30e3, 
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=t_tip_settle+ramp_time+hold_time+t_data_buffer, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         **kwargs)
        self.display_progress = display_progress
        self.t_tip_settle = t_tip_settle
        self.t_data_buffer = t_data_buffer
        self.t_seq_gap = t_seq_gap
        self.ramp_time = ramp_time 
        self.hold_time = hold_time
        self.P_min_dBm = P_min_dBm
        self.P_max_dBm = P_max_dBm
        self.P_hold_dBm = P_hold_dBm
        self.P_valon = P_valon
        V_min = var_attn_utils.get_vctrl(self.P_min_dBm, P_in=P_valon)
        V_max = var_attn_utils.get_vctrl(self.P_max_dBm, P_in=P_valon)
        V_hold = var_attn_utils.get_vctrl(self.P_hold_dBm, P_in=P_valon)
        self.AWG_sweep = RigolAWGSweep(v_min=V_min, v_max=V_max, v_hold=V_hold, 
                                       flat_time=t_tip_settle, ramp_time=ramp_time, 
                                       hold_time=hold_time, AWG=AWG, channel=AWG.channel)
        self.Valon.set_power(self.P_valon)
        self.Valon.output_on()

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.trigger_on()
        self.AWG_sweep.run()
        time.sleep(self.AWG_sweep.running_time)
        self.SSA.get_trace()
        time.sleep(1)
        all_meas = []
        all_t_sweep_start = [] 
        all_t_sweep_end = []
        all_t_tip_on = [] 
        all_t_tip_off = []
        all_t_seq_end = []

        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            t0 = time.time()
            self.FEtip_PSU.set_voltage(self.V_on)
            # time.sleep(self.t_tip_settle)
            t1 = time.time()
            self.AWG_sweep.run()
            time.sleep(self.AWG_sweep.running_time)
            t2 = time.time()
            self.FEtip_PSU.set_voltage(self.V_off)
            t3 = time.time()
            time.sleep(self.t_data_buffer)
            data = self.SSA.get_trace()
            t4 = time.time()
            
            all_meas.append(data)
            all_t_sweep_start.append(t1)
            all_t_sweep_end.append(t2)
            all_t_tip_on.append(t0)
            all_t_tip_off.append(t3)
            all_t_seq_end.append(t4)
            time.sleep(self.t_seq_gap)

        self.FEtip_PSU.ramp_down(0)
        self.SSA.trigger_off()

        return {'all_meas': np.array(all_meas), 't_sweep_start': np.array(all_t_sweep_start)+self.t_tip_settle, 
                't_sweep_end': np.array(all_t_sweep_end), 't_tip_on': np.array(all_t_tip_on),
                't_tip_off': np.array(all_t_tip_off), 't_seq_end': np.array(all_t_seq_end)}
    

class RFPowerSweepAWGTipSwitched(RFPowerSweepVolLinTipSwitched): 
    def __init__(self, name='RFPowerSweepAWGTipSwitched', saving_dir=None,
                 P_min_dBm=-10, P_max_dBm =0, P_hold_dBm=-9,
                 ramp_time=0.1, hold_time=0.05,
                 AWG=None, P_valon=5.0,
                 t_data_buffer=1, t_seq_gap=0.5, N_avg=5, 
                 V_on=1400, V_off=700, t_tip_settle=3,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW = 30e3, 
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=ramp_time+hold_time*2+t_data_buffer, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         **kwargs)
        self.display_progress = display_progress
        self.t_tip_settle = t_tip_settle
        self.t_data_buffer = t_data_buffer
        self.t_seq_gap = t_seq_gap
        self.ramp_time = ramp_time 
        self.hold_time = hold_time
        self.P_min_dBm = P_min_dBm
        self.P_max_dBm = P_max_dBm
        self.P_hold_dBm = P_hold_dBm
        self.P_valon = P_valon
        V_min = var_attn_utils.get_vctrl(self.P_min_dBm, P_in=P_valon)
        V_max = var_attn_utils.get_vctrl(self.P_max_dBm, P_in=P_valon)
        V_hold = var_attn_utils.get_vctrl(self.P_hold_dBm, P_in=P_valon)
        self.AWG_sweep = RigolAWGSweep(v_min=V_min, v_max=V_max, v_hold=V_hold, 
                                       flat_time=hold_time, ramp_time=ramp_time, 
                                       hold_time=hold_time, AWG=AWG, channel=AWG.channel)
        self.Valon.set_power(self.P_valon)
        self.Valon.output_on()
        

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.trigger_on()
        self.AWG_sweep.run()
        time.sleep(self.AWG_sweep.running_time)
        self.SSA.get_trace()
        time.sleep(1)
        all_meas = []
        all_t_sweep_start = [] 
        all_t_sweep_end = []
        all_t_tip_on = [] 
        all_t_tip_off = []
        all_t_seq_end = []

        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            t0 = time.time()
            self.FEtip_PSU.set_voltage(self.V_on)
            time.sleep(self.t_tip_settle)
            t1 = time.time()
            self.FEtip_PSU.set_voltage(self.V_off)
            t2 = time.time()
            self.AWG_sweep.run()
            time.sleep(self.AWG_sweep.running_time)           
            t3 = time.time()
            time.sleep(self.t_data_buffer)
            data = self.SSA.get_trace()
            t4 = time.time()
            
            all_meas.append(data)
            all_t_sweep_start.append(t2)
            all_t_sweep_end.append(t3)
            all_t_tip_on.append(t0)
            all_t_tip_off.append(t1)
            all_t_seq_end.append(t4)
            time.sleep(self.t_seq_gap)

        self.FEtip_PSU.ramp_down(0)
        self.SSA.trigger_off()

        return {'all_meas': np.array(all_meas), 't_sweep_start': np.array(all_t_sweep_start),#+self.t_tip_settle, 
                't_sweep_end': np.array(all_t_sweep_end), 't_tip_on': np.array(all_t_tip_on),
                't_tip_off': np.array(all_t_tip_off), 't_seq_end': np.array(all_t_seq_end)}
    
class RFPowerConstSweepTipSwitched(RFPowerSweepVolLinear): 
    def __init__(self, name='RFPowerConstSweepTipSwitched', P_valon_hold=0,
                 valon_settle=2, loading_time=3, measurement_time=5,
                 P_min_dBm=1, P_max_dBm=10, R=50, steps=800,  
                 V_on=1400, V_off=700,
                 SSA_freq_center = None, SSA_freq_span = None,
                 SSA_RBW = None, SSA_SWT=None,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 N_inner=3, N_outer=3, 
                 display_progress="outer",
                 **kwargs): 
        super().__init__(name, Valon=Valon, P_min_dBm=P_min_dBm, steps=steps,
                         P_max_dBm=P_max_dBm, R=R, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, SSA_RBW=SSA_RBW, SSA_SWT=SSA_SWT, SSA=SSA, FEtip_PSU=FEtip_PSU,
                         **kwargs)
        self.display_progress = display_progress
        self.disable_outer = self.display_progress != "outer"
        self.disable_pscan = self.display_progress != "Pscan"
        self.disable_inner = self.display_progress != "inner"

        self.valon_settle = valon_settle
        # check if measurement_time >= loading_time. If not, throw an error
        if measurement_time < loading_time:
            raise ValueError(f"Invalid times: measurement_time ({measurement_time}) must be greater than or equal to loading_time ({loading_time}).")
        self.loading_time = loading_time
        self.measurement_time = measurement_time
        self.N_inner = N_inner
        self.N_outer = N_outer
        self.P_valon_hold = P_valon_hold
        self.Valon.set_power(self.P_valon_hold)
        self.Valon.output_on()
        

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.trigger_off()

        # self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        voltage_scan = np.linspace(self.V_min, self.V_max, self.steps)
        dBm_scan = np.around(Vp_to_dBm(voltage_scan, R=self.R), 5)
        # all_meas = []
        # all_t_meas= [] 
        results = {'dBm_scan': dBm_scan}
        self.Valon.set_power(dBm_scan[0])
        self.Valon.output_on()
        # print(f"Starting experiment. SA will sweep continuously for {self.measurement_time}. Loading time is {self.loading_time}.")
        for repetition in self.progress_bar(range(self.N_outer), disable=self.disable_outer, desc='>>> Outer averaging'):
            # outer_meas = []
            # outer_t_meas = []
            for dBm_Valon in self.progress_bar(dBm_scan, disable=self.disable_pscan, desc='>>> Power scans'):
                # inner_meas = []
                # inner_t_meas = []
                self.Valon.set_power(dBm_Valon)
                self.Valon.output_on()
                sleep(self.valon_settle)
                for average in self.progress_bar(range(self.N_inner), disable=self.disable_inner, desc='>>> Inner repetitions'):
                    self.FEtip_PSU.set_voltage(self.V_on)
                    tip_is_on = True
                    t0 = time.perf_counter()
                    all_inner = [] 
                    all_t_inner = []
                    while True:
                        now = time.perf_counter() - t0
                        # 1. Check if it's time to stop the entire experiment
                        if now >= self.measurement_time:
                            break
                        # 2. Check if it's time to turn the tip off (Only execute once)
                        if tip_is_on and now >= self.loading_time:
                            self.FEtip_PSU.set_voltage(self.V_off)
                            tip_is_on = False # Prevents this block from triggering again
                        # 3. Capture SA Data immediately (This blocks only for the hardware sweep duration)
                        sweep_start_time = time.perf_counter() - t0
                        data = self.SSA.get_full_trace()
                        sweep_stop_time = time.perf_counter() - t0
                        all_inner.append(data)
                        all_t_inner.append((sweep_start_time, sweep_stop_time))
                    # print(inner_t_meas)
                    # inner_meas.append()
                    # inner_t_meas.append()
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_data'] = np.concatenate(all_inner)
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_time'] = np.concatenate([np.linspace(all_t_inner[i][0], all_t_inner[i][1], len(all_inner[i])) for i in range(len(all_t_inner))])
            # all_meas.append(outer_meas)
            # all_t_meas.append(outer_t_meas)
        self.FEtip_PSU.ramp_down(0)
        self.Valon.set_power(0)
        self.Valon.output_off()
        return results ## fixing inhomogeneous array issue
    

class RFPowerSweepOnOffTipOff(RFPowerSweepVolLinTipSwitched): 
    def __init__(self, name='RFPowerSweepOnOff', saving_dir=None,
                 P_load=10, P_detect=5, V_on=1050, V_off=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=t_data, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         **kwargs)
        self.display_progress = display_progress
        self.P_load = P_load 
        self.P_detect = P_detect 
        self.t_load = t_load 
        self.t_data = t_data 
        self.t_rest = t_rest 

    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.get_full_trace()
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            # self.SSA.get_full_trace()
            self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.set_power(self.P_detect)
            data = self.SSA.get_full_trace()
            all_meas.append(data)
            # self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}
    

class RFPowerSweepOnOff(RFPowerSweepVolLinTipSwitched): 
    def __init__(self, name='RFPowerSweepOnOff', saving_dir=None,
                 P_load=10, P_detect=5, V_on=1050, V_off=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32, SSA_SWT=None,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=t_data if SSA_SWT is None else SSA_SWT, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         **kwargs)
        self.display_progress = display_progress
        self.P_load = P_load 
        self.P_detect = P_detect 
        self.t_load = t_load 
        self.t_data = t_data 
        self.t_rest = t_rest 

    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.get_full_trace()
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            # self.SSA.get_full_trace()
            self.Valon.set_power(self.P_detect)
            data = self.SSA.get_full_trace()
            all_meas.append(data)
            self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}
    

class RFPowerSweepAWGArbRamp(RFPowerSweepVolLinTipSwitched): 
    def __init__(self, name='RFPowerSweepAWGArbRamp', saving_dir=None,
                 P_min_dBm=-10, P_max_dBm =0, P_start_dBm=-9, P_hold_dBm=-9,
                 ramp_time=0.1, hold_time=0.05,
                 AWG=None, P_valon=5.0,
                 t_data_buffer=1, t_seq_gap=0.5, N_avg=5, 
                 V_on=1400, V_off=700, t_tip_settle=3,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW = 30e3, 
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=t_tip_settle+ramp_time+hold_time+t_data_buffer, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         **kwargs)
        self.display_progress = display_progress
        self.t_tip_settle = t_tip_settle
        self.t_data_buffer = t_data_buffer
        self.t_seq_gap = t_seq_gap
        self.ramp_time = ramp_time 
        self.hold_time = hold_time
        self.flat_time = t_tip_settle
        self.P_min_dBm = P_min_dBm
        self.P_max_dBm = P_max_dBm
        self.P_flat_dBm = P_start_dBm
        self.P_hold_dBm = P_hold_dBm
        self.P_valon = P_valon
        V_min = var_attn_utils.get_vctrl(self.P_min_dBm, P_in=P_valon)
        V_max = var_attn_utils.get_vctrl(self.P_max_dBm, P_in=P_valon)
        V_flat = var_attn_utils.get_vctrl(self.P_flat_dBm, P_in=P_valon)
        V_hold = var_attn_utils.get_vctrl(self.P_hold_dBm, P_in=P_valon)
        self.AWG_sweep = RigolAWGSweep(v_min=V_min, v_max=V_max, v_hold=V_hold, v_flat=V_flat,
                                       flat_time=t_tip_settle, ramp_time=ramp_time, 
                                       hold_time=hold_time, AWG=AWG, channel=AWG.channel)
        self.Valon.set_power(self.P_valon)
        self.Valon.output_on()

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up(self.V_off)
        self.SSA.clear_averaging()
        self.SSA.trigger_on()
        self.AWG_sweep.run()
        time.sleep(self.AWG_sweep.running_time)
        self.SSA.get_trace()
        time.sleep(1)
        all_meas = []
        all_t_sweep_start = [] 
        all_t_sweep_end = []
        all_t_tip_on = [] 
        all_t_tip_off = []
        all_t_seq_end = []

        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            t0 = time.time()
            self.FEtip_PSU.set_voltage(self.V_on)
            # time.sleep(self.t_tip_settle)
            t1 = time.time()
            self.AWG_sweep.run()
            time.sleep(self.AWG_sweep.running_time)
            t2 = time.time()
            # self.FEtip_PSU.set_voltage(self.V_off)
            t3 = time.time()
            time.sleep(self.t_data_buffer)
            data = self.SSA.get_trace()
            t4 = time.time()
            
            all_meas.append(data)
            all_t_sweep_start.append(t1)
            all_t_sweep_end.append(t2)
            all_t_tip_on.append(t0)
            all_t_tip_off.append(t3)
            all_t_seq_end.append(t4)
            time.sleep(self.t_seq_gap)

        self.FEtip_PSU.ramp_down(0)
        self.SSA.trigger_off()

        return {'all_meas': np.array(all_meas), 't_sweep_start': np.array(all_t_sweep_start)+self.t_tip_settle, 
                't_sweep_end': np.array(all_t_sweep_end), 't_tip_on': np.array(all_t_tip_on),
                't_tip_off': np.array(all_t_tip_off), 't_seq_end': np.array(all_t_seq_end)}
    
class RFPowerSweepOnOffDifferential(RFPowerSweepOnOff): 
    def __init__(self, name='RFPowerSweepOnOffDifferential', saving_dir=None,
                 P_load=10, P_detect=5, V_on=800, V_off=500, V_pos=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         P_load=P_load, P_detect=P_detect, 
                         t_load=t_load, t_data=t_data, t_rest=t_rest,
                         N_avg=N_avg, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         display_progress=display_progress,
                         **kwargs)
        self.V_pos = V_pos

    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_off)
        
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            # self.SSA.get_full_trace()
            # self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.set_power(self.P_detect)
            data = self.SSA.get_full_trace()
            all_meas.append(data)
            
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch('pos', 0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}
    
class RFPowerSweepOnOffDifferentialTipSwitched(RFPowerSweepOnOffDifferential):
    def __init__(self, name='RFPowerSweepOnOffDifferentialTipSwitched', saving_dir=None,
                 P_load=10, P_detect=5, V_on=800, V_off=500, V_pos=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         P_load=P_load, P_detect=P_detect, 
                         t_load=t_load, t_data=t_data, t_rest=t_rest,
                         N_avg=N_avg, V_on=V_on, V_off=V_off, V_pos=V_pos,
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         display_progress=display_progress,
                         **kwargs)
    
    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_off)
        
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):

            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            # self.SSA.get_full_trace()
            self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.set_power(self.P_detect)
            data = self.SSA.get_full_trace()
            all_meas.append(data)
            
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch('pos', 0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}
    

class RFPowerConstSweepDifferentialTipSwitched(RFPowerSweepVolLinear): 
    def __init__(self, name='RFPowerConstSweepDifferentialTipSwitched', P_valon_hold=0,
                 valon_settle=2, loading_time=3, measurement_time=5,
                 P_min_dBm=1, P_max_dBm=10, R=50, steps=800,  
                 V_on=1400, V_off=700, V_pos=600,
                 SSA_freq_center = None, SSA_freq_span = None,
                 SSA_RBW = None, SSA_SWT=None,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 N_inner=3, N_outer=3, 
                 display_progress="outer",
                 **kwargs): 
        super().__init__(name, Valon=Valon, P_min_dBm=P_min_dBm, steps=steps,
                         P_max_dBm=P_max_dBm, R=R, V_on=V_on, V_off=V_off, 
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, SSA_RBW=SSA_RBW, SSA_SWT=SSA_SWT, SSA=SSA, FEtip_PSU=FEtip_PSU,
                         **kwargs)
        self.display_progress = display_progress
        self.disable_outer = self.display_progress != "outer"
        self.disable_pscan = self.display_progress != "Pscan"
        self.disable_inner = self.display_progress != "inner"

        self.valon_settle = valon_settle
        # check if measurement_time >= loading_time. If not, throw an error
        if measurement_time < loading_time:
            raise ValueError(f"Invalid times: measurement_time ({measurement_time}) must be greater than or equal to loading_time ({loading_time}).")
        self.loading_time = loading_time
        self.measurement_time = measurement_time
        self.N_inner = N_inner
        self.N_outer = N_outer
        self.V_pos = V_pos
        self.P_valon_hold = P_valon_hold
        self.Valon.set_power(self.P_valon_hold)
        self.Valon.output_on()
        

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_on)
        self.SSA.clear_averaging()
        self.SSA.trigger_off()

        # self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        voltage_scan = np.linspace(self.V_min, self.V_max, self.steps)
        dBm_scan = np.around(Vp_to_dBm(voltage_scan, R=self.R), 5)
        # all_meas = []
        # all_t_meas= [] 
        results = {'dBm_scan': dBm_scan}
        self.Valon.set_power(dBm_scan[0])
        self.Valon.output_on()
        
        # print(f"Starting experiment. SA will sweep continuously for {self.measurement_time}. Loading time is {self.loading_time}.")
        for repetition in self.progress_bar(range(self.N_outer), disable=self.disable_outer, desc='>>> Outer averaging'):
            # outer_meas = []
            # outer_t_meas = []
            for dBm_Valon in self.progress_bar(dBm_scan, disable=self.disable_pscan, desc='>>> Power scans'):
                # inner_meas = []
                # inner_t_meas = []
                self.Valon.set_power(dBm_Valon)
                self.Valon.output_on()
                time.sleep(self.valon_settle)
                for average in self.progress_bar(range(self.N_inner), disable=self.disable_inner, desc='>>> Inner repetitions'):
                    
                    tip_is_on = True
                    t0 = time.perf_counter()
                    all_inner = [] 
                    all_t_inner = []
                    while True:
                        now = time.perf_counter() - t0
                        # 1. Check if it's time to stop the entire experiment
                        if now >= self.measurement_time:
                            break
                        # 2. Check if it's time to turn the tip off (Only execute once)
                        if tip_is_on and now >= self.loading_time:
                            self.FEtip_PSU.set_voltage(self.V_off)
                            tip_is_on = False # Prevents this block from triggering again
                        # 3. Capture SA Data immediately (This blocks only for the hardware sweep duration)
                        sweep_start_time = time.perf_counter() - t0
                        data = self.SSA.get_full_trace()
                        sweep_stop_time = time.perf_counter() - t0
                        all_inner.append(data)
                        all_t_inner.append((sweep_start_time, sweep_stop_time))
                    # print(inner_t_meas)
                    # inner_meas.append()
                    # inner_t_meas.append()
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_data'] = np.concatenate(all_inner)
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_time'] = np.concatenate([np.linspace(all_t_inner[i][0], all_t_inner[i][1], len(all_inner[i])) for i in range(len(all_t_inner))])
            # all_meas.append(outer_meas)
            # all_t_meas.append(outer_t_meas)
        self.FEtip_PSU.ramp_down(0)
        self.Valon.set_power(0)
        self.Valon.output_off()
        return results ## fixing inhomogeneous array issue
    

class RFPowerConstSweepDifferential(RFPowerConstSweepDifferentialTipSwitched): 
    def __init__(self, name='RFPowerConstSweepDifferential', P_valon_hold=0,
                 valon_settle=2, loading_time=3, measurement_time=5,
                 P_min_dBm=1, P_max_dBm=10, R=50, steps=800,  
                 V_on=1400, V_off=700, V_pos=600,
                 SSA_freq_center = None, SSA_freq_span = None,
                 SSA_RBW = None, SSA_SWT=None,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 N_inner=3, N_outer=3, 
                 display_progress="outer",
                 **kwargs): 
        super().__init__(name=name, P_valon_hold=P_valon_hold,
                 valon_settle=valon_settle, loading_time=loading_time, measurement_time=measurement_time,
                 P_min_dBm=P_min_dBm, P_max_dBm=P_max_dBm, R=R, steps=steps,  
                 V_on=V_on, V_off=V_off, V_pos=V_pos,
                 SSA_freq_center = SSA_freq_center, SSA_freq_span = SSA_freq_span,
                 SSA_RBW = SSA_RBW, SSA_SWT=SSA_SWT,
                 Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU,
                 N_inner=N_inner, N_outer=N_outer, 
                 display_progress=display_progress,
                 **kwargs)
        

    def run(self):
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_on)
        self.SSA.clear_averaging()
        self.SSA.trigger_off()

        # self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        voltage_scan = np.linspace(self.V_min, self.V_max, self.steps)
        dBm_scan = np.around(Vp_to_dBm(voltage_scan, R=self.R), 5)
        # all_meas = []
        # all_t_meas= [] 
        results = {'dBm_scan': dBm_scan}
        self.Valon.set_power(dBm_scan[0])
        self.Valon.output_on()
        
        # print(f"Starting experiment. SA will sweep continuously for {self.measurement_time}. Loading time is {self.loading_time}.")
        for repetition in self.progress_bar(range(self.N_outer), disable=self.disable_outer, desc='>>> Outer averaging'):
            # outer_meas = []
            # outer_t_meas = []
            for dBm_Valon in self.progress_bar(dBm_scan, disable=self.disable_pscan, desc='>>> Power scans'):
                # inner_meas = []
                # inner_t_meas = []
                self.Valon.set_power(dBm_Valon)
                self.Valon.output_on()
                time.sleep(self.valon_settle)
                for average in self.progress_bar(range(self.N_inner), disable=self.disable_inner, desc='>>> Inner repetitions'):
                    
                    tip_is_on = True
                    t0 = time.perf_counter()
                    all_inner = [] 
                    all_t_inner = []
                    while True:
                        now = time.perf_counter() - t0
                        # 1. Check if it's time to stop the entire experiment
                        if now >= self.measurement_time:
                            break
                        # 2. Check if it's time to turn the tip off (Only execute once)
                        # if tip_is_on and now >= self.loading_time:
                        #     self.FEtip_PSU.set_voltage(self.V_off)
                        #     tip_is_on = False # Prevents this block from triggering again
                        # # 3. Capture SA Data immediately (This blocks only for the hardware sweep duration)
                        sweep_start_time = time.perf_counter() - t0
                        data = self.SSA.get_full_trace()
                        sweep_stop_time = time.perf_counter() - t0
                        all_inner.append(data)
                        all_t_inner.append((sweep_start_time, sweep_stop_time))
                    # print(inner_t_meas)
                    # inner_meas.append()
                    # inner_t_meas.append()
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_data'] = np.concatenate(all_inner)
                    results[f'dBm{dBm_Valon:.5f}_average{average}_repetition{repetition}_time'] = np.concatenate([np.linspace(all_t_inner[i][0], all_t_inner[i][1], len(all_inner[i])) for i in range(len(all_t_inner))])
            # all_meas.append(outer_meas)
            # all_t_meas.append(outer_t_meas)
        self.FEtip_PSU.ramp_down(0)
        self.Valon.set_power(0)
        self.Valon.output_off()
        return results ## fixing inhomogeneous array issue
    

class RFPowerSweepRevivalDifferential(RFPowerSweepOnOffDifferentialTipSwitched):
    def __init__(self, name='RFPowerSweepRevivalDifferential', saving_dir=None,
                 P_load=10, P_detect=5, V_on=800, V_off=500, V_pos=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         P_load=P_load, P_detect=P_detect, 
                         t_load=t_load, t_data=t_data, t_rest=t_rest,
                         N_avg=N_avg, V_on=V_on, V_off=V_off, V_pos=V_pos,
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, 
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         display_progress=display_progress,
                         **kwargs)
    
    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_off)
        
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)
        P_seq = [self.P_detect, self.P_load, self.P_detect]
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):
            loc_data = []
            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            for P in P_seq:
                self.SSA.clear_averaging() 
                self.Valon.set_power(P)
                data = self.SSA.get_full_trace()
                loc_data.extend(data)
            all_meas.append(loc_data)
            
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch('pos', 0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}
    

class RFPowerSweepDifferentialNAcquisition(RFPowerSweepRevivalDifferential):
    def __init__(self, name='RFPowerSweepDifferentialNAcquisition', saving_dir=None,
                 P_load=10, P_detect=5, V_on=800, V_off=500, V_pos=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32, SSA_SWT=0.1,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         P_load=P_load, P_detect=P_detect, 
                         t_load=t_load, t_data=t_data, t_rest=t_rest,
                         N_avg=N_avg, V_on=V_on, V_off=V_off, V_pos=V_pos,
                         SSA_freq_center=SSA_freq_center, SSA_freq_span=SSA_freq_span, 
                         SSA_RBW=SSA_RBW, SSA_SWT=SSA_SWT,
                         Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU, 
                         display_progress=display_progress,
                         **kwargs)
    
    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_off)
        
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)
        N_data = int(self.t_data/0.1)
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):
            loc_data = []
            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            self.Valon.set_power(self.P_detect)
            for _ in range(N_data):
                data = self.SSA.get_full_trace()
                loc_data.extend(data)
            all_meas.append(loc_data)
            
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch('pos', 0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}


class RFPowerSweepDifferentialNAcquisitionTipSwitched(RFPowerSweepDifferentialNAcquisition):
    def __init__(self, name='RFPowerSweepDifferentialNAcquisitionTipSwitched', saving_dir=None,
                 P_load=10, P_detect=5, V_on=800, V_off=500, V_pos=600,
                 t_load=10, t_data=1, t_rest=1,
                 SSA_freq_center=None, SSA_freq_span = None,
                 SSA_RBW=30e3, N_avg=32, SSA_SWT=0.1,
                 Valon=None, SSA=None, FEtip_PSU=None,
                 display_progress=True,
                 **kwargs): 
        super().__init__(name, saving_dir=saving_dir, 
                         P_load=P_load, P_detect=P_detect, 
                         V_on=V_on, V_off=V_off, V_pos=V_pos,
                        t_load=t_load, t_data=t_data, t_rest=t_rest,
                        SSA_freq_center=SSA_freq_center, SSA_freq_span = SSA_freq_span,
                        SSA_RBW=SSA_RBW, N_avg=N_avg, SSA_SWT=SSA_SWT,
                        Valon=Valon, SSA=SSA, FEtip_PSU=FEtip_PSU,
                        display_progress=display_progress,
                         **kwargs)
    
    def run(self):
        self.SSA.select_mode('SA')
        self.SSA_init(self.SSA_freq_center, self.SSA_freq_span, self.SSA_RBW, self.SSA_SWT)
        self.FEtip_PSU.ramp_up_ch('pos', self.V_pos)
        self.FEtip_PSU.ramp_up(self.V_off)
        
        self.SSA.clear_averaging()
        data = self.SSA.get_full_trace()
        self.SSA.set_div_scale(5) 
        self.SSA.set_ref_level(max(data)+25)
        N_data = int(self.t_data/0.1)
        all_meas = []
        for _ in self.progress_bar(range(self.N_avg), disable=(not self.display_progress)):
            loc_data = []
            self.FEtip_PSU.set_voltage(self.V_on)
            self.Valon.output_on() 
            self.Valon.set_power(self.P_load)
            time.sleep(self.t_load)
            self.SSA.clear_averaging() 
            self.FEtip_PSU.set_voltage(self.V_off)
            self.Valon.set_power(self.P_detect)
            for _ in range(N_data):
                data = self.SSA.get_full_trace()
                loc_data.extend(data)
            all_meas.append(loc_data)
            
            self.Valon.output_off()
            time.sleep(self.t_rest)

        self.FEtip_PSU.ramp_down(0)
        self.FEtip_PSU.ramp_down_ch('pos', 0)
        self.Valon.output_off()
        self.SSA.clear_averaging()

        return {'all_meas': np.array(all_meas)}