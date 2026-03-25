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