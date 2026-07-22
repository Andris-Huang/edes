# from lakeshore import Model240, Model240InputParameter, Model240CurveHeader
import pynanovna
from pynanovna.utils import stream_from_csv
from pynanovna.vis import plot, polar
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
import datetime
import os
from scipy.optimize import curve_fit
import warnings
import requests
import json
from edes.modules.detection.detection_utils import plot, plot_ax, plot_ax_errbar, plot_errbar, big_plt_font
big_plt_font()
warnings.filterwarnings('once')

# vna = pynanovna.VNA()
def S21(f,f0,Q,Qc,a,t):
    #constant phase offset
    phase1 = np.exp(1j*0)
    #frequency dependent phase
    phase2 = np.exp(-2*np.pi*1j*f*0)
    #ideal resonator
    F = (Q/Qc)/(1+2*1j*Q*(f-f0)/f0)

    #real and imaginary components
    S21_complex = phase1*phase2*F
    S21_re = np.real(S21_complex)
    S21_im = np.imag(S21_complex)

    return np.concatenate((S21_re,S21_im))

def S11(f,f0,Q,Qc,a,t):
    #constant phase offset
    phase1 = np.exp(1j*0)
    #frequency dependent phase
    phase2 = np.exp(-2*np.pi*1j*f*0)
    #ideal resonator
    F = (1 - 2*(Q/Qc)/(1+2*1j*Q*(f-f0)/f0))

    #real and imaginary components
    S21_complex = phase1*phase2*F
    S21_re = np.real(S21_complex)
    S21_im = np.imag(S21_complex)

    return np.concatenate((S21_re,S21_im))

def vec_S21(S21_concat):
    l = int(len(S21_concat)/2)
    r = S21_concat[:l]
    i = S21_concat[l:]
    return [r,i]

def vec_S11(S11_concat):
    l = int(len(S11_concat)/2)
    r = S11_concat[:l]
    i = S11_concat[l:]
    return [r,i]

def mag_S21(re_s21, im_s21):
    return (np.sqrt(re_s21**2+im_s21**2))

def phase_S21(re_s21, im_s21):
    return np.arctan(im_s21/re_s21)

def get_Qi(Q, Qc):
    return 1 / (1/Q - 1/Qc)

def fit_S21(s2p_file, pi = [270e6,500,1000,0,0], C='blue', slabel='', start=185e6, stop=205e6):
  filepath = s2p_file
  df = pd.read_csv(filepath,skiprows=6,names=['S11', 'S21', 'freq'])

  freqs_np = np.array(df.freq)
  ReS21 = np.array(df['S21']).astype(complex).real
  ImS21 = np.array(df['S21']).astype(complex).imag

  magS21 = np.sqrt(ReS21**2+ImS21**2)

  

  max_freq = (freqs_np[np.where(magS21==np.max(magS21))])

  start = 250e6  #max_freq[0]-3e6
  stop= 300e6  #max_freq[0]+3e6


  min_index = np.min(np.where(freqs_np>=start)[0])
  max_index = np.max(np.where(freqs_np<=stop)[0])


  reS21_np = np.array(ReS21)[min_index:max_index]
  imS21_np = np.array(ImS21)[min_index:max_index]
  freqs=freqs_np[min_index:max_index]
  pi[0] = max_freq[0] 
    

  S21_data = np.concatenate((reS21_np,imS21_np))

  params, pcov = curve_fit(S21, freqs, S21_data,p0=pi)

  S21_Concat = S21(freqs,params[0],params[1],params[2],0,0)#,params[3],params[4])
  S21_Vec = vec_S21(S21_Concat)

  #plt.plot(freqs/1e6, mag_S21(S21_Vec[0],S21_Vec[1]),'.',color='r', label=f'Q = {round(params[1],2)}')

  S21_mags = np.sqrt(S21_Vec[0]**2+S21_Vec[1]**2)

  plt.plot(freqs/1e6,S21_mags,color=C, label=slabel+f'Q = {round(params[1],2)}')
  plt.xlabel('frequency (MHZ)')
  plt.ylabel('mag(S21)')
  plt.legend()

  
  return params

def fit_S11(s2p_file, pi = [227e6,200,100,0,0], C='blue', slabel='', start=185e6, stop=205e6):
  filepath = s2p_file
  #df = pd.read_csv(filepath,skiprows=6,names=['S11', 'S21', 'freq'])
  df = pd.read_csv(filepath)

  freqs_np = np.array(df['freq'], dtype=np.float64)
  ReS11 = np.array(df['S11'], dtype=np.complex128).real
  ImS11 = np.array(df['S11'], dtype=np.complex128).imag

  print(np.shape(ReS11))

  magS11 = np.sqrt(ReS11**2+ImS11**2)

  

  max_freq = (freqs_np[np.where(magS11==np.max(magS11))])

  start = 210e6  #max_freq[0]-3e6
  stop= 250e6  #max_freq[0]+3e6


  min_index = np.min(np.where(freqs_np>=start)[0])
  max_index = np.max(np.where(freqs_np<=stop)[0])


  reS11_np = np.array(ReS11)[min_index:max_index]
  imS11_np = np.array(ImS11)[min_index:max_index]
  freqs=freqs_np[min_index:max_index]
  pi[0] = max_freq[0] 
    

  S11_data = np.concatenate((reS11_np,imS11_np))

  params, pcov = curve_fit(S11, freqs, S11_data,p0=pi)

  S11_Concat = S11(freqs,params[0],params[1],params[2],0,0)#,params[3],params[4])
  #S11_Concat = S11(freqs,pi[0],pi[1],pi[2],0,0)#,params[3],params[4])
  S11_Vec = vec_S11(S11_Concat)

  #plt.plot(freqs/1e6, mag_S21(S21_Vec[0],S21_Vec[1]),'.',color='r', label=f'Q = {round(params[1],2)}')

  S11_mags = np.sqrt(S11_Vec[0]**2+S11_Vec[1]**2)

  plt.plot(freqs/1e6,S11_mags,color=C, label=slabel+f'Q = {round(params[1],2)}')
  plt.plot(freqs/1e6, magS11[min_index:max_index])
  plt.xlabel('frequency (MHZ)')
  plt.ylabel('mag(S11)')
  plt.legend()
  return params

#initialize VNA
def vna_sweep(vna, sweep_range,num_points,path,default=101):

    # default = 101

    start = sweep_range[0]
    stop = sweep_range[1]
    point_range = stop-start


    if num_points>default:
        S11 = []
        S21 = []
    
        num_sweeps = int((num_points - num_points%default)/default + 1) #number of sweeps required 
        # print(f'doing {num_sweeps-1} sweeps of {default} points and 1 sweep of  {num_points%default} to achieve {num_points} points')
        sub_sweep = point_range/num_sweeps 

        
        
        S11_sweeps = []
        S21_sweeps = []
        freq_sweeps = []
        
        for i in range(1, num_sweeps):
            freq = np.linspace(start+(i-1)*sub_sweep, start+i*sub_sweep, default)
            # print(min(freq), max(freq), len(freq))
            vna.set_sweep(min(freq), max(freq), len(freq))
            stream = vna.stream()
            data0, data1, freq = vna.sweep()
            
            S11_sweeps.append(data0)
            S21_sweeps.append(data1)
            freq_sweeps.append(freq)

        freq = np.linspace(start+(num_sweeps-1)*sub_sweep, start+num_sweeps*sub_sweep,  num_points%default)
        # print(min(freq), max(freq), len(freq))
        
        vna.set_sweep(min(freq), max(freq), len(freq))
        stream = vna.stream()
        time.sleep(1)
        data0, data1, freq = vna.sweep()
        time.sleep(1)
        S11_sweeps.append(data0)
        S21_sweeps.append(data1)
        freq_sweeps.append(freq)

        S11_data = np.concatenate(S11_sweeps)
        S21_data = np.concatenate(S21_sweeps)
        freq_data = np.concatenate(freq_sweeps)

        data = {'S11': S11_data, 'S21': S21_data, 'freq': freq_data }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    else:
        #vna sweep
        vna.set_sweep(start, stop, num_points)
        stream = vna.stream()
        S11_data, S21_data, freq_data = vna.sweep()

        data = {'S11': S11_data, 'S21': S21_data, 'freq': freq_data }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        #filename = path
        #vna.stream_to_csv(filename, 1)

    return S11_data, S21_data, freq_data

def vna_plot(path):
    #plot VNA data
    df = pd.read_csv(path,skiprows = 7,names=['S11','S21','freq'])
    
    S11 = np.array(df['S11'])
    S21 = np.array(df['S21'])
    freq = np.array(df['freq'])

    '''
    #find the row that doesnt have data
    bad_index = np.where(S11=='sweepnumber: ')[0][0]
    '''
    bad_index = -1
    #get rid of bad row1 
    S11_clean = np.array(S11[bad_index+1:]).astype(complex)
    S21_clean = np.array(S21[bad_index+1:]).astype(complex)
    freq_clean = np.array(freq[bad_index+1:])
    S11_mag_db = 20*np.log10(np.abs(S11_clean))
    S21_mag_db = 20*np.log10(np.abs(S21_clean))
    
    plt.plot(freq_clean*1e-6,S11_mag_db,'.',label='S11', color='blue')
    plt.plot(freq_clean*1e-6,S11_mag_db,alpha=0.2, color='blue')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('S11 (dB)')
    '''
    res = (freq_clean[np.where(S21_mag_db==max(S21_mag_db))])*1e-6
    plt.plot(freq_clean*1e-6,S21_mag_db,'.',label='S21', color='blue')
    #plt.plot(freq_clean*1e-6,S21_mag_db,alpha=0.2, color='blue')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('S21 (dB)')
    '''
    res = (freq_clean[np.where(S21_mag_db==max(S21_mag_db))])*1e-6
    plt.title(path)
    plt.legend()

def take_vna_sweep(filename, folder_path, f_begin, f_end, N_points=2000): 
    file_path = os.path.join(folder_path,filename)
    S11_0, S21_0, freq = vna_sweep(vna, (f_begin, f_end), N_points, file_path)
    return S11_0, S21_0, freq

def query_temp(stage, T_to_channels = {'40K': 1, '4K': 2, '0.4K': 5}): 
    req = requests.get('http://169.254.168.2:5001/channel/measurement/latest', timeout=10)
    data = req.json()
    ch = T_to_channels[stage]
    while data['channel_nr'] != ch: 
        req = requests.get('http://169.254.168.2:5001/channel/measurement/latest', timeout=10)
        data = req.json()
    return data['temperature']

from resonator_tools import circuit 


def get_complex_Sparam(SSA, trace=1): 
    vna = SSA

    # Make sure VNA mode is active
    vna.write(":INSTrument:SELect VNA")
    vna.write(":INITiate:IMMediate")
    vna.write(f":CALCulate1:PARameter{trace}:SELect ")
    for form in ['MLOGarithmic', 'Phase']:
        vna.write(f":CALCulate1:FORMat {form}")
        data_str = vna.query(":CALCulate1:DATA:FDATa?")
        data = np.array(data_str.strip().split(','), dtype=float)
        # sleep(1)
        
        # f = data[0::2] 
        # data = data[1::2]
        if form == 'MLOGarithmic': 
            f, mag = data[0::2], data[1::2]
        else: 
            f2, phase = data[0::2], data[1::2]
    return f, 10**(mag/20)*np.exp(1j*phase/180*np.pi)

def plot_S11_fit(f, complex_data, plot=True, **kwargs): 
    port1 = circuit.reflection_port()
    port1.add_data(f,complex_data)
    port1.autofit(**kwargs)
    fit_results = port1.fitresults
    if plot:
        for i in ['fr', 'Qi', 'Qc', 'Ql']: 
            if i == 'fr':
                print(f'f0 = {fit_results[i]/1e6:.3f} +- {fit_results[i+"_err"]/1e6:.3f} MHz') 
            else: 
                print(f'{i} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
    if plot:
        plotall(port1, 'S11')
    else:
        return [fit_results[i] for i in ['fr', 'Qi', 'Qc', 'Ql']]
    

def plot_S21_fit(f, complex_data): 
    port1 = circuit.notch_port()
    port1.add_data(f,complex_data)
    port1.autofit()
    fit_results = port1.fitresults
    naming_map = {'Qi_no_corr': 'Qi', 'absQc': 'Qc'}
    for i in ['fr', 'Qi_no_corr', 'absQc', 'Ql']: 
        if i == 'fr':
            print(f'f0 = {fit_results[i]/1e9:.3f} +- {fit_results[i+"_err"]/1e9:.3f} GHz') 
        elif i in naming_map: 
            print(f'{naming_map[i]} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
        else: 
            print(f'{i} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
    plotall(port1, 'S21')

def measure_and_plot_S11(vna): 
    # Make sure VNA mode is active
    vna.write(":INSTrument:SELect VNA")
    vna.write(":INITiate:IMMediate")
     
    for form in ['MLOGarithmic', 'Phase']:
        vna.write(f":CALCulate1:FORMat {form}")
        data_str = vna.query(":CALCulate1:DATA:FDATa?")
        data = np.array(data_str.strip().split(','), dtype=float)
        sleep(3)
        
        # f = data[0::2] 
        # data = data[1::2]
        if form == 'MLOGarithmic': 
            f, mag = data[0::2], data[1::2]
        else: 
            f2, phase = data[0::2], data[1::2]
    fig, ax = plt.subplots() 
    plot_ax(ax, f/1e9, mag, xlabel='Frequency (GHz)', ylabel='|S11| (dB)', label='Mag') 
    ax2 = ax.twinx()
    plot_ax(ax2, f/1e9, phase, c='C1', ylabel=r'arg(S11) (deg)', label='Phase')
    ax2.grid(False)
    ax.legend() 
    ax2.legend()
    return f, mag, phase

def measure_and_plot_S11_S21(vna): 

    # Make sure VNA mode is active
    vna.write(":INSTrument:SELect VNA")
    vna.write(":INITiate:IMMediate")
    
    vna.write(":CALCulate1:PARameter1:SELect ")
    vna.write(":CALCulate1:FORMat MLOGarithmic")
    
    data_str = vna.query(":CALCulate1:DATA:FDATa?")
    data = np.array(data_str.strip().split(','), dtype=float)
    f, S11_mag = data[0::2], data[1::2]
    vna.write(":CALCulate1:PARameter2:SELect ")
    vna.write(":CALCulate1:FORMat MLOGarithmic")
    data_str = vna.query(":CALCulate1:DATA:FDATa?")
    data = np.array(data_str.strip().split(','), dtype=float)
    f, S21_mag = data[0::2], data[1::2]

    pplot(f/1e9, S11_mag, label='S11')
    plot(f/1e9, S21_mag, xlabel='f (GHz)', ylabel='|S| (dB)', label='S21')
    plt.legend()
    return f, S11_mag, S21_mag

def plotall(self, S_name='S'):
    fig, ax = plt.subplots(ncols=3, figsize=(12,3))
    real = self.z_data_raw.real
    imag = self.z_data_raw.imag
    real2 = self.z_data_sim.real
    imag2 = self.z_data_sim.imag
    plot_ax(ax[0], real,imag,label='data')
    plot_ax(ax[0], real2,imag2,label='fit', xlabel=f'Re({S_name})', ylabel=f'Im({S_name})')
    ax[0].legend()
    plot_ax(ax[1], self.f_data*1e-9,20.*np.log10(np.absolute(self.z_data_raw)),label='data')
    plot_ax(ax[1], self.f_data*1e-9,20.*np.log10(np.absolute(self.z_data_sim)),label='fit', xlabel='f (GHz)', ylabel=f'|{S_name}| (dB)')
    ax[1].legend()
    plot_ax(ax[2], self.f_data*1e-9,np.angle(self.z_data_raw),label='data')
    plot_ax(ax[2], self.f_data*1e-9,np.angle(self.z_data_sim),label='fit', xlabel='f (GHz)', ylabel=f'arg(|{S_name}|)')
    ax[2].legend()
    plt.tight_layout()
    plt.show()


import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Watt2dBm(x):
	'''
	converts from units of watts to dBm
	'''
	return 10.*np.log10(x*1000.)
	
def dBm2Watt(x):
	'''
	converts from units of watts to dBm
	'''
	return 10**(x/10.) /1000.	

class plotting(object):
	'''
	some helper functions for plotting
	'''
	def plotall(self):
		real = self.z_data_raw.real
		imag = self.z_data_raw.imag
		real2 = self.z_data_sim.real
		imag2 = self.z_data_sim.imag
		plt.subplot(221)
		plt.plot(real,imag,label='rawdata')
		plt.plot(real2,imag2,label='fit')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.subplot(222)
		plt.plot(self.f_data*1e-9,20.*np.log10(np.absolute(self.z_data_raw)),label='rawdata')
		plt.plot(self.f_data*1e-9,20.*np.log10(np.absolute(self.z_data_sim)),label='fit')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21| (dB)')
		plt.legend()
		plt.subplot(223)
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_sim),label='fit')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(S21)')
		plt.legend()
		plt.show()
		
	def plotcalibrateddata(self):
		real = self.z_data.real
		imag = self.z_data.imag
		plt.subplot(221)
		plt.plot(real,imag,label='rawdata')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.subplot(222)
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21|')
		plt.legend()
		plt.subplot(223)
		plt.plot(self.f_data*1e-9,np.angle(self.z_data),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(S21)')
		plt.legend()
		plt.show()
		
	def plotrawdata(self):
		real = self.z_data_raw.real
		imag = self.z_data_raw.imag
		plt.subplot(221)
		plt.plot(real,imag,label='rawdata')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.subplot(222)
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data_raw),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21|')
		plt.legend()
		plt.subplot(223)
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(S21)')
		plt.legend()
		plt.show()

class save_load(object):
	'''
	procedures for loading and saving data used by other classes
	'''
	def _ConvToCompl(self,x,y,dtype):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		if dtype=='realimag':
			return x+1j*y
		elif dtype=='linmagphaserad':
			return x*np.exp(1j*y)
		elif dtype=='dBmagphaserad':
			return 10**(x/20.)*np.exp(1j*y)
		elif dtype=='linmagphasedeg':
			return x*np.exp(1j*y/180.*np.pi)
		elif dtype=='dBmagphasedeg':
			return 10**(x/20.)*np.exp(1j*y/180.*np.pi)	 
		else: warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)
	
	def add_data(self,f_data,z_data):
		self.f_data = np.array(f_data)
		self.z_data_raw = np.array(z_data)
		
	def cut_data(self,f1,f2):
		def findpos(f_data,val):
			pos = 0
			for i in range(len(f_data)):
				if f_data[i]<val: pos=i
			return pos
		pos1 = findpos(self.f_data,f1)
		pos2 = findpos(self.f_data,f2)
		self.f_data = self.f_data[pos1:pos2]
		self.z_data_raw = self.z_data_raw[pos1:pos2]
		
	def add_fromtxt(self,fname,dtype,header_rows,usecols=(0,1,2),fdata_unit=1.,delimiter=None):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		data = np.loadtxt(fname,usecols=usecols,skiprows=header_rows,delimiter=delimiter)
		self.f_data = data[:,0]*fdata_unit
		self.z_data_raw = self._ConvToCompl(data[:,1],data[:,2],dtype=dtype)
		
	def add_fromhdf():
		pass
	
	def add_froms2p(self,fname,y1_col,y2_col,dtype,fdata_unit=1.,delimiter=None):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		if dtype == 'dBmagphasedeg' or dtype == 'linmagphasedeg':
			phase_conversion = 1./180.*np.pi
		else: 
			phase_conversion = 1.
		f = open(fname)
		lines = f.readlines()
		f.close()
		z_data_raw = []
		f_data = []
		if dtype=='realimag':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!")) :
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					z_data_raw.append(complex(float(lineinfo[y1_col]),float(lineinfo[y2_col])))
		elif dtype=='linmagphaserad' or dtype=='linmagphasedeg':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					z_data_raw.append(float(lineinfo[y1_col])*np.exp( complex(0.,phase_conversion*float(lineinfo[y2_col]))))
		elif dtype=='dBmagphaserad' or dtype=='dBmagphasedeg':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					linamp = 10**(float(lineinfo[y1_col])/20.)
					z_data_raw.append(linamp*np.exp( complex(0.,phase_conversion*float(lineinfo[y2_col]))))
		else:
			warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)
		self.f_data = np.array(f_data)
		self.z_data_raw = np.array(z_data_raw)
	def myadd_froms2p(self,fname,dtype,fdata_unit=1.):
		'''
		dtype = 'S11', 'S21'
		'''
		if dtype != 'S11' and dtype != 'S21':
			warnings.warn("Undefined input type! Use 'S11' or 'S21'.", SyntaxWarning)
		data=pd.read_csv(fname)
		self.f_data = np.array(data['freq'])*fdata_unit
		self.z_data_raw = np.array(data[dtype]).astype(complex)
		
	def save_fitresults(self,fname):
		pass
	

