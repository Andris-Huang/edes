import pyvisa as visa
from time import sleep, time
import matplotlib.pyplot as plt
import numpy as np
import importlib
from edes.utils.file_handling import load_lib



class Instrument: 
    def __init__(self, address, name='Instrument', log_callback=None):
        self.name = name
        self.log_callback = log_callback if log_callback else print
        self.rm = visa.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(address)
            self.instrument.timeout = 5000 # 2 seconds
            self.log_callback(f">>> Connected to {self.instrument.query('*IDN?')}")
        except Exception as e:
            self.log_callback(f">>> ERROR connecting to {self.name} at {address}: {e}")

    def query(self, command):
        return self.instrument.query(command)

    def write(self, command):
        self.instrument.write(command)
    
    def close(self):
        self.instrument.close() 
    
    def __str__(self):
        return self.name


class RigolDP832A(Instrument):
    def __init__(self, address, name='RigolDP832A', **kwargs):
        super().__init__(address, name=name, **kwargs)
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
    def __init__(self, address, name='SiglentSPD3303X_E', **kwargs): 
        super().__init__(address, name=name, **kwargs) 

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
    def __init__(self, address='ASRL/dev/ttyUSB1::INSTR', name='Valon', freq=1452e6, power=0, log_callback=None):
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
        self.log_callback = log_callback if log_callback else print
        self.log_callback(f">>> Connected to Valon at {address} with frequency {freq} Hz and power {power} dBm.")

    def output_on(self):
        self.valon.buffer_amplifiers_enabled(True)
    
    def output_off(self):
        self.valon.buffer_amplifiers_enabled(False) 
    
    def set_frequency(self, freq_hz):
        self.valon.frequency(freq_hz) 

    def set_power(self, power_dbm):
        self.valon.power(power_dbm) 
    
    def set_voltage(self, voltage, R=50, digits=5): 
        self.set_power(np.around(10 * np.log10((voltage**2) / (R*2)), digits))  # Convert voltage to power assuming 50 ohm load

    def query(self, command):
        self.log_callback(">>> Valon does not support query operations.")
        return

    def write(self, command):
        self.log_callback(">>> Valon does not support write operations.")
        return
    
    def close(self):
        self.output_off()

class TestValon: 
    """
    Fake Valon object for testing purposes
    """
    def __init__(self, address='', name='Valon', freq=1452e6, power=0, log_callback=None):
        self.log_callback = log_callback if log_callback else print
        self.log_callback(f">>> Connected to fake Valon")

    def output_on(self):
        return
    
    def output_off(self):
        return
    
    def set_frequency(self, freq_hz):
        return 

    def set_power(self, power_dbm):
        return 
    
    def set_voltage(self, voltage, R=50, digits=5): 
        return

    def query(self, command):
        return

    def write(self, command):
        return
    
    def close(self):
        self.output_off()


class PS350_viaDP832A(Instrument): 
    def __init__(self, address, name='PS350_viaDP832A', ch_ctrl=1, V_max=5000, V_offset=10, **kwargs): 
        self.name = name
        self.rigol = RigolDP832A(address, **kwargs) 
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


class PS350_viaDP832A_differential(PS350_viaDP832A): 
    def __init__(self, address, name='PS350_viaDP832A_2ch', default_ch_ctrl=1, ch_pos=2, ch_neg=1, 
                 Vpos_max=5000, Vpos_offset=10, Vneg_max=5000, Vneg_offset=10, **kwargs): 
        super().__init__(address, name=name, **kwargs)
        self.V1_max = Vneg_max
        self.V1_offset = Vneg_offset
        self.V2_max = Vpos_max
        self.V2_offset = Vpos_offset
        self.ch1_ctrl = ch_neg
        self.ch2_ctrl = ch_pos
        self.rigol.output_off(self.ch1_ctrl)
        self.rigol.output_off(self.ch2_ctrl)
        self.V1_current = 0
        self.V2_current = 0
        self.default_ch_ctrl = default_ch_ctrl
        self.select_ch(default_ch_ctrl)
    
    def select_ch(self, ch): 
        if ch == 1 or type(ch) == str and 'neg' in ch.lower(): 
            self.ch_ctrl = self.ch1_ctrl
            self.V_max = self.V1_max
            self.V_offset = self.V1_offset
            self.V_current = self.V1_current
        elif ch == 2 or type(ch) == str and 'pos' in ch.lower(): 
            self.ch_ctrl = self.ch2_ctrl
            self.V_max = self.V2_max
            self.V_offset = self.V2_offset
            self.V_current = self.V2_current
        else: 
            raise ValueError("Invalid channel. Choose 1 or 2.")
    
    def sync_ch(self, ch): 
        if ch == 1 or type(ch) == str and 'neg' in ch.lower(): 
            self.V1_current = self.V_current
        elif ch == 2 or type(ch) == str and 'pos' in ch.lower(): 
            self.V2_current = self.V_current

    def set_voltage_ch(self, ch, V_desired): 
        if ch == self.default_ch_ctrl:
            self.sync_ch(ch)
        self.select_ch(ch)
        self.set_voltage(V_desired)
        self.sync_ch(ch)
        self.select_ch(self.default_ch_ctrl)  # Return to default channel after setting voltage

    def ramp_up_ch(self, ch, V_target, delay=0.1, step=50):
        if ch == self.default_ch_ctrl:
            self.sync_ch(ch)
        self.select_ch(ch)
        self.ramp_up(V_target, delay, step)
        self.sync_ch(ch)
        self.select_ch(self.default_ch_ctrl)  # Return to default channel after ramping up

    def ramp_down_ch(self, ch, V_target, delay=0.1, step=50):
        if ch == self.default_ch_ctrl:
            self.sync_ch(ch)
        self.select_ch(ch)
        self.ramp_down(V_target, delay, step)
        self.sync_ch(ch)
        self.select_ch(self.default_ch_ctrl)  # Return to default channel after ramping down

    def output_off_all(self): 
        self.select_ch(self.ch1_ctrl)
        self.output_off()
        self.sync_ch(self.ch1_ctrl)
        self.select_ch(self.ch2_ctrl)
        self.output_off()
        self.sync_ch(self.ch2_ctrl)
        self.select_ch(self.default_ch_ctrl)  # Return to default channel after ramping up


class SSA3032X_R(Instrument): 
    def __init__(self, address, name='SSA3032X_R',
                 freq_center=198.016e6, freq_span=0, 
                 RBW_auto=0, VBW_auto=1, SWT_auto=1, VBW_RBW_rat=1, 
                 RBW=1, VBW=1, SWT=1, N_avg=32, **kwargs): 
        super().__init__(address, name=name, **kwargs)
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

        self.default_freq_center = freq_center 
        self.default_freq_span = freq_span 
        self.default_RBW_AUTO = RBW_auto 
        self.default_VBW_AUTO = VBW_auto
        self.default_SWT_AUTO = SWT_auto
        self.default_VBW_RBW_RAT = VBW_RBW_rat
        self.default_RBW = RBW
        self.default_VBW = VBW
        self.default_SWT = SWT
        self.default_N_AVG = N_avg

        self.initialize_SSA()

    def initialize_SSA(self):
        SSA = self.instrument

        # Configure frequency
        SSA.write("SENS:FREQ:CENT " + str(self.freq_center) + " Hz")
        SSA.write("SENS:FREQ:SPAN " + str(self.freq_span) + " Hz")

        # Configure bandwidths and sweep time
        SSA.write("SENS:BWID:AUTO " + str(self.RBW_AUTO))
        SSA.write("SENS:BWID:VID:AUTO " + str(self.VBW_AUTO))
        SSA.write("SENS:SWE:TIME:AUTO " + str(self.SWT_AUTO))
        SSA.write("SENS:BWID:VID " + str(self.VBW) + " Hz")
        SSA.write("SENS:BWID " + str(self.RBW) + " Hz")
        SSA.write("SENS:BWID:VID:RAT " + str(self.VBW_RBW_RAT))
        SSA.write("SENS:SWE:TIME " + str(self.SWT) + " s")
        SSA.write(f':AVERage:TRACe1:COUNt {self.N_AVG}')
        SSA.write(':AVERage:TRACe1:STATe ON')

        # Set sweep mode to FFT
        SSA.write(":SWEep:MODE FFT")
    
    def re_init(self, freq_center, span, RBW, SWT, VBW=None):
        self.set_freq_center(freq_center)
        self.set_freq_span(span)
        self.set_RBW(RBW)
        VBW = VBW if VBW is not None else self.VBW_RBW_RAT * RBW
        self.set_VBW(VBW)
        self.set_SWT(SWT)
        self.clear_averaging()
        data = self.get_full_trace()
        self.set_div_scale(5) 
        self.set_ref_level(max(data)+25)

    def re_init_RTSA(self, freq_center, span, RBW, SWT, VBW=None):
        self.set_freq_center(freq_center)
        self.set_freq_span(span)
        self.set_RBW(RBW)
        # VBW = VBW if VBW is not None else self.VBW_RBW_RAT * RBW
        # self.set_VBW(VBW)
        self.set_SWT(SWT)
        self.continuous_off()
        data = self.get_PvT_trace()
        self.set_div_scale(5) 
        self.set_ref_level(max(data)+25)
        self.continuous_on()

    def set_freq_center(self, freq_center):
        self.freq_center = freq_center
        self.instrument.write("SENS:FREQ:CENT " + str(self.freq_center) + " Hz")
    
    def set_freq_span(self, freq_span):
        self.freq_span = freq_span
        self.instrument.write("SENS:FREQ:SPAN " + str(self.freq_span) + " Hz")
    
    def set_RBW(self, RBW):
        self.RBW = RBW
        self.instrument.write("SENS:BWID " + str(self.RBW) + " Hz") 
    
    def set_VBW(self, VBW):
        self.VBW = VBW
        self.instrument.write("SENS:BWID:VID " + str(self.VBW) + " Hz")
    
    def set_SWT(self, SWT): 
        self.SWT = SWT
        self.instrument.write("SENS:SWE:TIME " + str(self.SWT) + " s")

    def set_N_avg(self, N_avg):
        self.N_AVG = N_avg
        self.instrument.write(f':AVERage:TRACe1:COUNt {self.N_AVG}')
    
    def measure_freq_spectrum(self, trace=1, freq_center=None, freq_span=None, 
                              N_avg=None, RBW=None):
        SSA = self.instrument
        if freq_center is not None:
            self.freq_center = freq_center
            SSA.write("SENS:FREQ:CENT " + str(self.freq_center) + " Hz")
        if freq_span is not None:
            self.freq_span = freq_span
            SSA.write("SENS:FREQ:SPAN " + str(self.freq_span) + " Hz")
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
    
    def measure_VNA_spectrum(self, trace=1, freq_center=None, freq_span=None, 
                              N_avg=None, RBW=None):
        SSA = self.instrument
        if freq_center is not None:
            self.freq_center = freq_center
            SSA.write("SENS:FREQ:CENT " + str(self.freq_center) + " Hz")
        if freq_span is not None:
            self.freq_span = freq_span
            SSA.write("SENS:FREQ:SPAN " + str(self.freq_span) + " Hz")
        if N_avg is not None:
            self.N_AVG = N_avg
            SSA.write(f':AVERage:TRACe1:COUNt {self.N_AVG}')
        self.clear_averaging()
        sleep(self.SWT + 0.5)  # Wait for the sweep to complete (SWT + buffer time)
        data_str_C = SSA.query(f":TRACe{trace}:DATA?")
        data_arr_C = np.array([float(val) for val in data_str_C.split(',')]) 
        
        return data_arr_C[::2], data_arr_C[1::2]  # Return frequencies and magnitudes separately

    def set_param(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            self.log_callback(f">>> Parameter {param_name} not found in SSA3032X_R, creating new one.")
            setattr(self, param_name, value)
    
    def clear_averaging(self, trace=1):
        SSA = self.instrument
        SSA.write(":INITiate:CONTinuous ON")
        SSA.write(f":AVERage:TRAC{trace}:CLEar")
        # SSA.query(f":TRACe{trace}:DATA?")

    def get_trace(self, trace=1):
        SSA = self.instrument
        SSA.write(":INITiate:CONTinuous OFF")
        data_str_raw = SSA.query(f":TRACe{trace}:DATA?")
        data_arr = np.array([float(val) for val in data_str_raw.split(',')]) 
        SSA.write(":INITiate:CONTinuous ON")
        return data_arr
    
    def get_full_trace(self, trace=1):
        SSA = self.instrument
        
        # 1. Put the analyzer in single sweep mode
        # This stops it from continuously overwriting the trace buffer
        SSA.write(":INITiate:CONTinuous OFF") 
        
        # 2. Trigger a single sweep
        SSA.write(":INITiate:IMMediate")
        
        # 3. Wait for the sweep to finish
        # *OPC? blocks the script and returns '1' only when the preceding operations (the sweep) are complete
        SSA.query("*OPC?") 
        
        # 4. Fetch the trace data
        data_str_raw = SSA.query(f":TRACe{trace}:DATA?")
        
        # Clean and convert the data to a NumPy array
        # (Some analyzers return a header like '#41024...', so you may need to strip that if your current code fails)
        data_arr = np.array([float(val) for val in data_str_raw.split(',')]) 
        
        # 5. Optional: Return the analyzer to continuous sweep mode
        # SSA.write(":INITiate:CONTinuous ON")
        
        return data_arr
    
    def get_PvT_trace(self, trace=1):
        SSA = self.instrument
        SSA.write(":INITiate:CONTinuous OFF") 
        # 2. Trigger a single sweep
        SSA.write(":INITiate:IMMediate")
        sleep(self.SWT+10e-6)
        # 4. Fetch the trace data
        data_str_raw = SSA.query(f":TRACe:PVT?")
        
        # Clean and convert the data to a NumPy array
        data_arr = np.array([float(val) for val in data_str_raw.split(',')]) 
        
        # 5. Optional: Return the analyzer to continuous sweep mode
        SSA.write(":INITiate:CONTinuous ON")
        
        return data_arr
    
    def auto_scale(self): 
        self.write(f':DISPlay:WINDow:TRACe:Y:SCALe:AUTO')
    
    def select_mode(self, mode='VNA'):
        """
        Select the mode of operation.
        
        Parameters
        --- 
        * mode : [str]
            Select between 'VNA' and 'SA'
        """ 
        self.write(f':INSTrument {mode}')

    def set_ref_level(self, ref_level): 
        """
        Set the reference level for SA measurement
        """
        self.write(f':DISPlay:WINDow:TRACe:Y:RLEVel {ref_level} DBM')

    def set_div_scale(self, div):
        """
        Sets the power scale per division grid on screen
        """
        self.write(f':DISPlay:WINDow:TRACe:Y:PDIVision {div} dB')
    
    def preamp_on(self):
        """
        Turn on preamplifier in SA.
        """
        self.write(':POWer:GAIN ON')
    
    def preamp_off(self): 
        """
        Turn of preamplifier in SA.
        """
        self.write(':POWer:GAIN OFF')
    
    def set_SA_mode(self, mode='FFT'): 
        """
        Set the swept mode of SA to be either FFT or Sweep.
        """
        if mode == 'FFT': 
            self.write(':SWEep:MODE FFT')
        else: 
            self.write(':SWEep:MODE SWEep')

    def trigger_on(self): 
        self.write(":TRIGger:SOURce EXTernal")
    
    def trigger_off(self):
        self.write("*CLS")
        self.write(":TRIGger:SOURce IMMediate")
        self.write(":INITiate:CONTinuous ON")
        self.write(":INITiate:IMMediate")
    
    def continuous_on(self):
        self.write(":INITiate:CONTinuous ON")

    def continuous_off(self):
        self.write(":INITiate:CONTinuous OFF")

    def clear(self):
        self.write("*CLS")


class Keithley2100(Instrument): 
    def __init__(self, address, name='Keithley2100', **kwargs):
        super().__init__(address, name=name, **kwargs)

    def measure_V(self): 
        return float(self.instrument.query("MEAS:VOLT:DC?"))


class Agilent34461A(Instrument): 
    def __init__(self, address, name='Agilent34461A', **kwargs):
        super().__init__(address, name=name, **kwargs)

    def read(self): 
        raw_data = self.query(":READ?") 
        return float(raw_data.split(',')[0])

    def measure_V(self): 
        # self.write(":SENS:FUNC 'VOLT'") 
        return self.read()

class RigolDG4062(Instrument):
    def __init__(self, address, name='RigolDG4062', channel=1, **kwargs):
        super().__init__(address, name=name, **kwargs)
        self.write("*CLS")
        self.channel = channel

    def ramp_config(self, channel, freq, ampl, offset, symmetry):
        self.write(f":SOURce{channel}:FUNCtion RAMP")
        self.write(f":SOURce{channel}:FUNCtion:RAMP:SYMMetry {symmetry}")
        self.write(f":SOURce{channel}:FREQuency {freq}")
        self.write(f":SOURce{channel}:VOLTage:AMPLitude {ampl}")
        self.write(f":SOURce{channel}:VOLTage:OFFSet {offset}")

    def configure_ramp_sweep(self, channel=1, start_v=0.0, stop_v=2.0, sweep_time=0.1):
        """Builds the hardware configuration for a linear ramp sweep."""
        ch = f":SOURce{channel}"
        
        # Calculate amplitude and offset from requested min/max voltages
        ampl = abs(stop_v - start_v)
        offset = (stop_v + start_v) / 2.0
        freq = 1.0 / sweep_time # 1 cycle per sweep duration
        symmetry = 100

        # Set up a pure ramp-up (sawtooth) profile
        self.ramp_config(channel, freq, ampl, offset, symmetry)
        
        # 2. Hardware Trigger Behavior Setup
        # Set the trigger source to internal (the AWG decides when the ramp starts)
        self.write(f"{ch}:SWEep:TRIGger:SOURce INTernal") 
        
        # Tell the rear BNC port to output a standard TTL pulse at the start of the sweep
        self.write(f":OUTPut{channel}:TRIGger ON")
        
        # Set the trigger edge polarity to Positive (Rising edge of the 5V TTL pulse)
        # Use "NEG" if your receiving device triggers on a falling edge
        self.write(f"{ch}:SWEep:TRIGger:SLOPe POSitive")

    def configure_burst_ramp(self, channel=1, start_v=0.0, stop_v=2.0, ramp_time=0.1, cycles=5):
        """
        Configures the AWG to output an exact number of ramp pulses upon receiving a trigger.
        :param channel: Output channel (1 or 2)
        :param start_v: Starting voltage of the ramp
        :param stop_v: Peak voltage of the ramp
        :param ramp_time: Duration of a SINGLE ramp cycle (seconds)
        :param cycles: Exact number of pulses to output per burst
        """
        ch = f":SOURce{channel}"
        # Calculate amplitude and offset for the ramp profile
        ampl = abs(stop_v - start_v)
        offset = (stop_v + start_v) / 2.0
        freq = 1.0 / ramp_time # Frequency of an individual ramp

        # 1. Base Waveform Configuration (Ramp up)
        self.write(f"{ch}:FUNCtion RAMP")
        self.write(f"{ch}:FUNCtion:RAMP:SYMMetry 100")
        self.write(f"{ch}:FREQuency {freq}")
        self.write(f"{ch}:VOLTage:AMPLitude {ampl}")
        self.write(f"{ch}:VOLTage:OFFSet {offset}")

        # 2. Configure Burst Mode
        self.write(f"{ch}:BURSt:STATe ON")
        self.write(f"{ch}:BURSt:MODE NCYCle")
        self.write(f"{ch}:BURSt:NCYCle {cycles}")

        # Set the idle voltage state between bursts to match your starting voltage
        # This keeps the output at start_v until the burst fires
        self.write(f"{ch}:BURSt:IDLE MINimum")

        # 3. FIX: Configure Trigger Source to MANual (Software Control)
        # This tells the instrument to wait silently until Python tells it to fire
        self.write(f"{ch}:BURSt:TRIGger:SOURce MANual") 
        
        # Keep rear panel trigger routing active so auxiliary hardware can listen
        self.write(f":OUTPut{channel}:TRIGger ON") 
        self.write(f"{ch}:BURSt:TRIGger:SLOPe POSitive")

    def configure_burst_custom_arb(self, channel=1, start_v=0.0, stop_v=2.0, hold_v=0.5, 
                                   flat_v=None,
                                flat_time=0.05, ramp_time=0.1, hold_time=0.05):
        """
        Configures the AWG to execute a custom 3-step arbitrary pulse sequence per cycle.
        Sequence: Flat (at start_v) -> Linear Ramp (to stop_v) -> Hold (at hold_v)
        """
        ch = f":SOURce{channel}"
        
        # 1. Calculate overall timing and frequency
        total_time = flat_time + ramp_time + hold_time
        freq = 1.0 / total_time  # Cycle repeating frequency
        
        # Generate a 1000-point vector divided proportionally among your blocks
        total_points = 1000
        flat_pts = int(total_points * (flat_time / total_time))
        ramp_pts = int(total_points * (ramp_time / total_time))
        hold_pts = total_points - flat_pts - ramp_pts  # Ensure exact sum to 1000
        
        voltages = []
        
        # Step A: Flat section at start voltage
        flat_v = hold_v if flat_v is None else flat_v
        voltages.extend([flat_v] * flat_pts)
        
        # Step B: Linear ramp from start to stop voltage
        if ramp_pts > 1:
            for i in range(ramp_pts):
                v = start_v + (stop_v - start_v) * (i / (ramp_pts - 1))
                voltages.append(v)
        elif ramp_pts == 1:
            voltages.append(stop_v)
            
        # Step C: Hold section at hold voltage
        voltages.extend([hold_v] * hold_pts)
        
        # 2. Normalize voltages between -1.0 and +1.0 for the RIGOL DAC hardware
        v_max = max(voltages)
        v_min = min(voltages)
        ampl = v_max - v_min
        if ampl == 0: 
            ampl = 1e-3  # Guard against flatline zero division
        offset = (v_max + v_min) / 2.0
        
        normalized_points = [(v - offset) / (ampl / 2.0) for v in voltages]
        csv_points = ",".join(f"{p:.4f}" for p in normalized_points)
        
        # 3. Upload data matrix to volatile memory 
        # (Tries both modern and legacy RIGOL SCPI variations for compatibility)
        try:
            self.write(f"{ch}:DATA VOLATILE, {csv_points}")
        except Exception:
            self.write(f":TRACe:DATA:DATA VOLATILE, {csv_points}")
            
        # Set active function to Arbitrary mode 
        # (Note: Use "ARBitrary" instead of "USER" if you own a newer DG800/DG900 series)
        try:
            self.write(f"{ch}:FUNCtion USER")
        except Exception:
            self.write(f"{ch}:FUNCtion ARBitrary")
        
        # 4. Bind hardware scaling windows and playback clock rate
        self.write(f"{ch}:FREQuency {freq}")
        self.write(f"{ch}:VOLTage:AMPLitude {ampl}")
        self.write(f"{ch}:VOLTage:OFFSet {offset}")
        
        # 5. Configure Burst Profile
        self.write(f"{ch}:BURSt:STATe ON")
        self.write(f"{ch}:BURSt:MODE NCYCle")
        self.write(f"{ch}:BURSt:NCYCle 1")
        
        # Tell the unit to freeze on the First Point (FPT) between burst triggers.
        # Because your first point maps directly to start_v, it will idle at start_v.
        self.write(f"{ch}:BURSt:IDLE FPT") 
        
        # 6. Safety Interlocking & Manual Software Triggering
        self.write(f"{ch}:BURSt:TRIGger:SOURce MANual")
        self.write(f":OUTPut{channel}:TRIGger ON")
        self.write(f"{ch}:BURSt:TRIGger:SLOPe POSitive")

    def fire_burst(self):
        self.write(f"*TRG")

    def set_output(self, channel=1, state=True):
        """Turns a channel output ON (True) or OFF (False)."""
        status = "ON" if state else "OFF"
        self.write(f":OUTPut{channel} {status}")

    def output_on(self, ch=None):
        if ch is None: 
            ch = self.channel
        self.set_output(channel=ch, state=True)
    
    def output_off(self, ch=None):
        if ch is None: 
            ch = self.channel
        self.set_output(channel=ch, state=False)


class Zotino(Instrument): 
    def __init__(self, address, name='Zotino', 
                 cfile=None, calibration_file=None ,voltage_map=None,
                 sweeper='/home/electron/artiq/src/artiq/applets/testGUI.py', 
                 log_callback=None): 
        self.log_callback = log_callback if log_callback else print
        self.log_callback(f">>> Connected to ARTIQ Zotino.")
        self.address = address
        self.name = name 
        self.cfile = cfile 
        self.calibration_file = calibration_file 
        self.voltage_map = voltage_map
        self.sweeper = load_lib(sweeper)

    def set_multipoles(self, Ex=0, Ey=0, Ez=0, U1=0, U2=0, U3=0, U4=0, U5=0): 
        self.voltage_map['Ex'] = Ex 
        self.voltage_map['Ey'] = Ey 
        self.voltage_map['Ez'] = Ez 
        self.voltage_map['U1'] = U1 
        self.voltage_map['U2'] = U2 
        self.voltage_map['U3'] = U3 
        self.voltage_map['U4'] = U4 
        self.voltage_map['U5'] = U5 
        self.sweeper.load_dac_prg(self.cfile, self.calibration_file, 
                                  list(self.voltage_map.values()))

    
    def ground_all(self): 
        for key in self.voltage_map: 
            self.voltage_map[key] = 0 
        self.sweeper.load_dac_prg(self.cfile, self.calibration_file, 
                                  list(self.voltage_map.values()))
    