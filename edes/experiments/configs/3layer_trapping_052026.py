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