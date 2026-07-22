import datetime

### Set default parameters
SSA_params = dict(freq_center=176.1e6,freq_span=1e6, 
                  RBW_auto=0, VBW_auto=1, SWT_auto=1, VBW_RBW_rat=1, 
                  RBW=1e3, VBW=1e3, SWT=1, N_avg=1)
Valon_params = dict(freq=1.465e9, power=0)
HV_PSU_params = dict(V_max=5000, V_offset=10)
AWG_params = dict(channel=1)

### DAC config
voltage_map = {'bl1':0, 'bl2':0, 'bl3':0, 'bl4':0, 'bl5':0,
            'br1':0, 'br2':0, 'br3':0, 'br4':0, 'br5':0,
            'tl1':0, 'tl2':0, 'tl3':0, 'tl4':0, 'tl5':0,
            'tr1':0, 'tr2':0, 'tr3':0, 'tr4':0, 'tr5':0, 
            'Ex':0, 'Ey':0, 'Ez':0, 'U1':0, 'U2':0, 'U3':0, 
            'U4':0, 'U5':0}
cfile = '/home/electron/artiq/experiment/control_files/cfile_0p25in_spacing_0p5um_grid_separate_electrodes.csv'
calibration_file = '/home/electron/artiq/experiment/zotino_calibration/zotino_calibration_He3_new_PS_without_filter_20260618_162512.txt'
DAC_params = {"voltage_map": voltage_map, 
              "cfile": cfile, 
              "calibration_file": calibration_file}


### Define devices
SSA = dict(addr='TCPIP::192.168.169.161::INSTR', device='SSA3032X_R', params=SSA_params)
Keithley = dict(addr='USB0::1510::8448::1243106::0::INSTR', device='Keithley2100', params={})
LNA_PSU = dict(addr='TCPIP::192.168.169.101::INSTR', device='SiglentSPD3303X_E', params={})
FEtip_PSU = dict(addr='USB0::6833::3601::DP8B260200018::0::INSTR', device='PS350_viaDP832A', params=HV_PSU_params)
Valon = dict(addr='ASRL/dev/ttyUSB0::INSTR', device='Valon', params=Valon_params)
Agilent = dict(addr='USB0::2391::7175::MY53200916::0::INSTR', device='Agilent34461A', params={})
AWG = dict(addr='TCPIP::192.168.169.104::INSTR', device='RigolDG4062', params=AWG_params)
DAC = dict(addr='TCPIP::192.168.169.118::INSTR', device='Zotino', params=DAC_params)


### Saving directory
saving_dir = f'/home/electron/data/experiment_{datetime.datetime.now().strftime("%m%d%Y")}'


