# from lakeshore import Model240, Model240InputParameter, Model240CurveHeader

import matplotlib
matplotlib.use('Qt5Agg')
import pynanovna
from pynanovna.utils import stream_from_csv
from pynanovna.vis import plot, polar
import pandas as pd
import numpy as np
import warnings
import time
import datetime
import os
from scipy.optimize import curve_fit
import warnings
from edes.modules.detection.detection_utils import plot, plot_ax, plot_ax_errbar, plot_errbar, big_plt_font
from edes.utils.circuits import frac_to_dB
from resonator_tools import circuit

import matplotlib.pyplot as plt

big_plt_font()
warnings.filterwarnings('once')

filename = 'PCB_3layer_trap_including_RFwires_after_circulator_20260824_155833.csv'
folder_path = r'edes/modules/VNA/PCB_trap_drive'
def plot_S11_fit(f, complex_data): 
    port1 = circuit.reflection_port()
    port1.add_data(f,complex_data)
    # port1.autofit(Ql_guess=20, fr_guess=f[np.argmin(np.abs(complex_data))])
    port1.GUIfit()
    fit_results = port1.fitresults
    for i in ['fr', 'Qi', 'Qc', 'Ql']: 
        if i == 'fr':
            print(f'f0 = {fit_results[i]/1e6:.3f} +- {fit_results[i+"_err"]/1e6:.3f} MHz') 
        else: 
            print(f'{i} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
    port1.plotall()

def plot_S21_fit(f, complex_data): 
    port1 = circuit.notch_port()
    port1.add_data(f,complex_data)
    port1.GUIfit()
    fit_results = port1.fitresults
    naming_map = {'Qi_no_corr': 'Qi', 'absQc': 'Qc'}
    for i in ['fr', 'Qi_no_corr', 'absQc', 'Ql']: 
        if i == 'fr':
            print(f'f0 = {fit_results[i]/1e9:.3f} +- {fit_results[i+"_err"]/1e9:.3f} GHz') 
        elif i in naming_map: 
            print(f'{naming_map[i]} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
        else: 
            print(f'{i} = {fit_results[i]:.3f} +- {fit_results[i+"_err"]:.3f}')
    port1.plotall()

## Quality factor fitting for each dip
# For S11 data
df = pd.read_csv(os.path.join(folder_path,filename),converters={
    'S11': complex,
    'S21': complex})
S11_0 = df['S11'].to_numpy()
S21_0 = df['S21'].to_numpy()
freq = df['freq'].to_numpy() 

plot_S21_fit(freq, S21_0)
plt.show()

