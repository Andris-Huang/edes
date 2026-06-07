# Components Design, Experimental Control, and Data Analysis Package for the Electron Project
## To install
```bash
pip install -e .
```

## Experimental control
### Modules
* `edes/experiments`
    - `analysis`: code needed for analzing data
    - `configs`: folder used to store all experimental configuration and default parameter files
    - `devices`: the devices programming functions
    - `notebooks/experimental_control`: this folder contains all the control runs for daily experiments
    - `notebooks/post_processing`: this folder contains the post processing notebooks for daily experiments
    - `sequences`: experimental sequences 
* `edes/modules`
    - Individual testing modules for different components, experimental instruments, and control sequences
* `edes/scripts`
    - General scripts needed to launch control GUI, if applicable.
* `edes/utils`
    - `circuits.py`: circuit related helper functions
    - `file_handling.py`: helper functions related to load/store files of different types, list file names, etc.
    - `fitting.py`: helper functions to fit data
    - `plotting.py`: predefined functions to plot data in common styles

### Creating a new experiment
1. First create a new configuration file in `edes/experiments/configs` folder, this file stores all the devices needed and their corresponding parameters. An example is shown below
    ```python
    import datetime

    ### Set default parameters
    SSA_params = dict(freq_center=198.016,freq_span=2, 
                    RBW_auto=0, VBW_auto=1, SWT_auto=1, VBW_RBW_rat=1, 
                    RBW=1, VBW=1, SWT=100, N_avg=32)
    Valon_params = dict(freq=1472114285, power=0)
    HV_PSU_params = dict(V_max=5000, V_offset=10)


    ### Define devices
    SSA = dict(addr='TCPIP::192.168.169.161::INSTR', device='SSA3032X_R', params=SSA_params)
    Keithley = dict(addr='USB0::1510::8448::1243106::0::INSTR', device='Keithley2100', params={})
    LNA_PSU = dict(addr='TCPIP::192.168.169.101::INSTR', device='SiglentSPD3303X_E', params={})
    FEtip_PSU = dict(addr='USB0::6833::3601::DP8B260200018::0::INSTR', device='PS350_viaDP832A', params=HV_PSU_params)
    Valon = dict(addr='ASRL/dev/ttyUSB0::INSTR', device='Valon', params=Valon_params)
    Agilent = dict(addr='USB0::2391::7175::MY53200916::0::INSTR', device='Agilent34461A', params={})


    ### Saving directory
    saving_dir = f'/home/electron/data/experiment_{datetime.datetime.now().strftime("%m%d%Y")}'
    ```

2. Add a new sequence in `edes/experiments/sequences` folder. If the sequence is basic, it can be added to `base.py` in the folder, otherwise a new `.py` file can be created to add the new sequences specific to a given experiment. An example sequence is shown below. Note that usually the only things to modify are the parameters needed for the sequence, the devices associated, and the actual sequence in the `run()` function.
    ```python
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
    ```

3. Add a new device if needed in `edes/experiments/devices`. Again, if it's a basic component, it can be added to `base.py`, otherwise create a new file. An example device control file is shown below.
    ```python
    class Keithley2100(Instrument): 
        def __init__(self, address, name='Keithley2100'):
            super().__init__(address, name=name)

        def measure_V(self): 
            return float(self.instrument.query("MEAS:VOLT:DC?"))
    ```

### Run experiment
1. First create a new notebook in `edes/experiments/notebooks/experimental_control`
2. Import the `Experiment` object and the desired sequences, as well as other common packages. Example shown below.
    ```python
    from edes.experiments import Experiment
    from edes.experiments.sequences.base import TipVoltageSweep
    from edes.utils.plotting import plot, plot_ax, big_plt_font, plot_ax_errbar, plot_errbar
    ```
3. Initialize an experiment given the config file name. Example below. Upon the excution of this code, all devices listed in the config file should be connected, otherwise an error would be printed.
    ```python
    exp = Experiment('3layer_trapping_052026')
    ```
4. Can list all the connected devices using `exp.list_devices()`. If a device is disconnected during the middle of experiment, can connect is back using `exp.load_device("<device_name>")`. Can use the individual device functions as `exp.<device>.<function>()`, and can close all deivces using `exp.close_all()`. Example below.
    ```python
    exp.list_devices()
    >>> ['Agilent', 'FEtip_PSU', 'Keithley', 'LNA_PSU', 'SSA']
    exp.Keithley.close()
    exp.load_device("Keithley")
    >>> Connected to KEITHLEY INSTRUMENTS INC.,MODEL 2100,1,01.07-01-01
    ```
5. Initialize a sequence and give them the parameters and devices needed. Example below.
    ```python
    tip_Vsweep = TipVoltageSweep(saving_dir=exp.saving_dir, 
                                V_start=1000, V_stop=3000, V_step=200, R=100e6, 
                                N_avg=3, t_PSU_settle=5, t_meas_delay=2, 
                                FEtip_PSU=exp.FEtip_PSU, multimeter=exp.Keithley)
    ```
6. Run sequence. Can either run without saving using `sequence.run()` or run with data saved using `sequence.run_save()`. Example below. Both function will return the result as a dictionary. Files are saved into the directory specified in the config file, otherwise the default saving directory is `logs` folder where the package is located.
    ```python
    file = tip_Vsweep.run_save()
    ```


## Data analysis
1. First create a new notebook in `edes/experiments/notebooks/post_processing`. Import the required helper functions. Example below.
    ```python
    from edes.utils.file_handling import load_h5_data, load_latest_data, load_latest_filename
    ```
2. Load the latest saved data using `file = load_latest_data()`. The function will return a dictonary with all the parameters and data saved. The data file itself is saved in `.h5` format and can be loaded separately too using `load_h5_data`.
3. Can list all the parameters and data using `file.items()`. Can also see the filename using `file['filename']`, as the filename is saved as a parameter upon loading.
