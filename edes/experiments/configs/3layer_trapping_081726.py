import datetime

### Set default parameters
SSA_params = dict(freq_center=175.8e6,freq_span=1e6, 
                  RBW_auto=0, VBW_auto=1, SWT_auto=1, VBW_RBW_rat=1, 
                  RBW=1e3, VBW=1e3, SWT=1, N_avg=1)
Valon_params = dict(freq=1463230000, power=0) #1465714285 1465200000
HV_PSU_params = dict(Vneg_max=2500, Vneg_offset=3, Vpos_max=5000, Vpos_offset=4, ch_neg=1, ch_pos=2, default_ch_ctrl='neg')
AWG_params = dict(channel=1)

### DAC config
voltage_map = {'bl1':0, 'bl2':0, 'bl3':0, 'bl4':0, 'bl5':0,
            'br1':0, 'br2':0, 'br3':0, 'br4':0, 'br5':0,
            'tl1':0, 'tl2':0, 'tl3':0, 'tl4':0, 'tl5':0,
            'tr1':0, 'tr2':0, 'tr3':0, 'tr4':0, 'tr5':0, 
            'Ex':0, 'Ey':0, 'Ez':0, 'U1':0, 'U2':0, 'U3':0, 
            'U4':0, 'U5':0}
cfile = '/home/electron/artiq/experiment/control_files/cfile_fused_DC_0p025in_FEP_Shield_Manual_0.3_Mesh_100um_0.5um_L_ROI_100um.csv'
calibration_file = '/home/electron/artiq/experiment/zotino_calibration/zotino_calibration_new_filter_board_500ms_20260819_143419.txt'
DAC_params = {"voltage_map": voltage_map, 
              "cfile": cfile, 
              "calibration_file": calibration_file}


### Define devices
SSA = dict(addr='TCPIP::192.168.169.161::INSTR', device='SSA3032X_R', params=SSA_params)
Keithley = dict(addr='USB0::1510::8448::1243106::0::INSTR', device='Keithley2100', params={})
LNA_PSU = dict(addr='TCPIP::192.168.169.101::INSTR', device='SiglentSPD3303X_E', params={})
FEtip_PSU = dict(addr='USB0::6833::3601::DP8B260200018::0::INSTR', device='PS350_viaDP832A_differential', params=HV_PSU_params)
Valon = dict(addr='ASRL/dev/ttyUSB0::INSTR', device='Valon', params=Valon_params)
Agilent = dict(addr='USB0::2391::7175::MY53200916::0::INSTR', device='Agilent34461A', params={})
AWG = dict(addr='TCPIP::192.168.169.104::INSTR', device='RigolDG4062', params=AWG_params)
DAC = dict(addr='TCPIP::192.168.169.118::INSTR', device='Zotino', params=DAC_params)



