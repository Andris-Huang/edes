import pyvisa as visa
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import importlib



class Instrument: 
    def __init__(self, address, name='Instrument'):
        self.name = name
        self.rm = visa.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(address)
            self.instrument.timeout = 5000 # 2 seconds
            print(f">>> Connected to {self.instrument.query('*IDN?')}")
        except Exception as e:
            print(f">>> ERROR connecting to instrument at {address}: {e}")

    def query(self, command):
        return self.instrument.query(command)

    def write(self, command):
        self.instrument.write(command)
    
    def close(self):
        self.instrument.close() 


class RigolDP832A(Instrument):
    def __init__(self, address, name='RigolDP832A'):
        super().__init__(address, name=name)
        # idn_response = self.query('*IDN?')
        # if 'RIGOL' in idn_response.upper() and 'DP832A' in idn_response.upper():
        #     print(f"Rigol DP832A found: {idn_response.strip()}")
        # else:
        #     raise Exception("Connected instrument is not a Rigol DP832A.")
    
    def select_channel(self, channel):
        self.write(f"INST:NSEL {channel}")

    def set_voltage(self, channel, voltage):
        self.select_channel(channel)
        self.write(f'VOLT {voltage}') 
    
    def set_current(self, channel, current):
        self.select_channel(channel)
        self.write(f'CURR {current}')

    def output_on(self, channel):   
        self.select_channel(channel)
        self.write(f'OUTP ON')    
    
    def output_off(self, channel):  
        self.select_channel(channel)
        self.write(f'OUTP OFF')  
    
    def measure_voltage(self, channel):
        self.select_channel(channel)
        response = self.query(f'MEAS:VOLT?')
        return float(response.strip())
    
    def measure_current(self, channel):
        self.select_channel(channel)
        response = self.query(f'MEAS:CURR?')
        return float(response.strip())  
    

class SiglentSPD3303X_E(Instrument): 
    def __init__(self, address, name='SiglentSPD3303X_E'): 
        super().__init__(address, name=name) 

    def select_channel(self, channel):
        self.write(f"INSTrument CH{channel}")

    def set_voltage(self, channel, voltage):
        self.select_channel(channel)
        self.write(f'VOLT {voltage}') 
    
    def set_current(self, channel, current):
        self.select_channel(channel)
        self.write(f'CURR {current}')

    def output_on(self, channel):   
        self.write(f"OUTPut CH{channel},ON")    
    
    def output_off(self, channel):  
        self.write(f"OUTPut CH{channel},OFF")
    
    def set_current_protection(self, max_current, channel): 
        self.write(f"LIMIT:CURRent {channel},{max_current}")

    def set_voltage_protection(self, max_voltage, channel): 
        self.write(f"LIMIT:VOLTage {channel},{max_voltage}")

    def measure_voltage(self, channel):
        self.select_channel(channel)
        response = self.query(f'MEAS:VOLT?')
        return float(response.strip()) 
        
    def measure_current(self, channel):
        self.select_channel(channel)
        response = self.query(f'MEAS:CURR?')
        return float(response.strip())  
    

class Valon(Instrument): 
    def __init__(self, address='ASRL/dev/ttyUSB1::INSTR', name='Valon', freq=1452e6, power=0):
        self.name = name
        spec = importlib.util.spec_from_file_location("device_lib", "/home/electron/Qcodes_contrib_drivers/src/qcodes_contrib_drivers/drivers/Valon/Valon_5015.py")
        device_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(device_lib)
        Valon5015 = device_lib.Valon5015
        self.valon = Valon5015(name=name, address=address)
        self.valon.frequency(freq)
        self.valon.offset(0)
        self.valon.power(power)
        self.valon.modulation_db(0)
        self.valon.modulation_frequency(1)
        self.valon.low_power_mode_enabled(True)
        self.valon.buffer_amplifiers_enabled(False)

    def output_on(self):
        self.valon.buffer_amplifiers_enabled(True)
    
    def output_off(self):
        self.valon.buffer_amplifiers_enabled(False) 
    
    def set_frequency(self, freq_hz):
        self.valon.frequency(freq_hz) 

    def set_power(self, power_dbm):
        self.valon.power(power_dbm) 
    
    def set_voltage(self, voltage): 
        self.set_power(10 * np.log10((voltage**2) / 100))  # Convert voltage to power assuming 50 ohm load

    def query(self, command):
        print(">>> Valon does not support query operations.")
        return

    def write(self, command):
        print(">>> Valon does not support write operations.")
        return
    
    def close(self):
        self.output_off()



class PS350_viaDP832A(Instrument): 
    def __init__(self, address, name='PS350_viaDP832A', ch_ctrl=1, V_max=5000, V_offset=10): 
        self.name = name
        self.rigol = RigolDP832A(address) 
        self.V_max = V_max 
        self.V_offset = V_offset
        self.ch_ctrl = ch_ctrl
        self.rigol.output_off(ch_ctrl)
        self.V_current = 0
    
    def close(self): 
        self.ramp_down(0)
        self.rigol.output_off(self.ch_ctrl)
        self.rigol.close()
        
    def set_voltage(self, V_desired): 
        self.rigol.set_voltage(self.ch_ctrl, max((V_desired-self.V_offset),0)/self.V_max*10)
        self.rigol.output_on(self.ch_ctrl)
        self.V_current = V_desired

    def ramp_up(self, V_target, delay=0.1, step=50): 
        V_current = self.V_current
        while V_current < V_target: 
            V_current += step
            if V_current > V_target: 
                V_current = V_target
            self.set_voltage(V_current)
            sleep(delay)

    def ramp_down(self, V_target, delay=0.1, step=50): 
        V_current = self.V_current
        while V_current > V_target: 
            V_current -= step
            if V_current < V_target: 
                V_current = V_target
            self.set_voltage(V_current)
            sleep(delay)

    def output_off(self): 
        self.set_voltage(0)
        self.rigol.output_off(self.ch_ctrl)


class SSA3032X_R(Instrument): 
    def __init__(self, address, name='SSA3032X_R',
                 freq_center=198.016e6, freq_span=0, 
                 RBW_auto=0, VBW_auto=1, SWT_auto=1, VBW_RBW_rat=1, 
                 RBW=1, VBW=1, SWT=100, N_avg=32): 
        super().__init__(address, name=name)
        SSA = self.instrument
        self.freq_center = freq_center
        self.freq_span = freq_span
        self.RBW_AUTO = RBW_auto
        self.VBW_AUTO = VBW_auto
        self.SWT_AUTO = SWT_auto
        self.VBW_RBW_RAT = VBW_RBW_rat
        self.RBW = RBW
        self.VBW = VBW
        self.SWT = SWT
        self.N_AVG = N_avg
        self.initialize_SSA()

    def initialize_SSA(self):
        SSA = self.instrument

        # Configure frequency
        SSA.write("SENS:FREQ:CENT " + str(self.freq_center) + " MHz")
        SSA.write("SENS:FREQ:SPAN " + str(self.freq_span) + " MHz")

        # Configure bandwidths and sweep time
        SSA.write("SENS:BWID:AUTO " + str(self.RBW_AUTO))
        SSA.write("SENS:BWID:VID:AUTO " + str(self.VBW_AUTO))
        SSA.write("SENS:SWE:TIME:AUTO " + str(self.SWT_AUTO))
        SSA.write("SENS:BWID:VID " + str(self.VBW) + " kHz")
        SSA.write("SENS:BWID " + str(self.RBW) + " kHz")
        SSA.write("SENS:BWID:VID:RAT " + str(self.VBW_RBW_RAT))
        SSA.write("SENS:SWE:TIME " + str(self.SWT) + " ms")
        SSA.write(f':AVERage:TRACe1:COUNt {self.N_AVG}')
        SSA.write(':AVERage:TRACe1:STATe ON')

        # Set sweep mode to FFT
        SSA.write(":SWEep:MODE FFT")
    
    def measure_freq_spectrum(self, trace=1, freq_center=None, freq_span=None, 
                              N_avg=None, RBW=None):
        SSA = self.instrument
        if freq_center is not None:
            self.freq_center = freq_center
            SSA.write("SENS:FREQ:CENT " + str(self.freq_center) + " MHz")
        if freq_span is not None:
            self.freq_span = freq_span
            SSA.write("SENS:FREQ:SPAN " + str(self.freq_span) + " MHz")
        if N_avg is not None:
            self.N_AVG = N_avg
            SSA.write(f':AVERage:TRACe1:COUNt {self.N_AVG}')
        self.clear_averaging()
        while int(SSA.query(f":AVERage:TRACe{trace}?")) < self.N_AVG: 
            continue
        data_str_C = SSA.query(f":TRACe{trace}:DATA?")
        data_arr_C = np.array([float(val) for val in data_str_C.split(',')]) 
        # Number of points in the trace
        num_points = len(data_arr_C)
        
        # Generate frequency axis in MHz
        freq_start = self.freq_center - self.freq_span / 2
        freq_stop  = self.freq_center + self.freq_span / 2
        freq_axis = np.linspace(freq_start, freq_stop, num_points)
        return freq_axis, data_arr_C

    def set_param(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            print(f">>> Parameter {param_name} not found in SSA3032X_R, creating new one.")
            setattr(self, param_name, value)
    
    def clear_averaging(self):
        SSA = self.instrument
        SSA.write(":AVERage:TRAC1:CLEar")
        SSA.query(":TRACe1:DATA?")

    def get_trace(self, trace=1):
        SSA = self.instrument
        data_str_raw = SSA.query(f":TRACe{trace}:DATA?")
        data_arr = np.array([float(val) for val in data_str_raw.split(',')]) 
        return data_arr


class Keithley2100(Instrument): 
    def __init__(self, address, name='Keithley2100'):
        super().__init__(address, name=name)

    def measure_V(self): 
        return float(self.instrument.query("MEAS:VOLT:DC?"))


class Agilent34461A(Instrument): 
    def __init__(self, address, name='Agilent34461A'):
        super().__init__(address, name=name)

    def read(self): 
        raw_data = self.query(":READ?") 
        return float(raw_data.split(',')[0])

    def measure_V(self): 
        # self.write(":SENS:FUNC 'VOLT'") 
        return self.read()