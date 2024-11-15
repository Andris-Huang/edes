import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.constants as spc

base = "e:\\ansys\\resonator_v1"
def plot_parallel_coord(param_labels, param_values, objective_label, objective_values, 
                        colorscale='thermal', reverse_scale=False, width=1000, height=600):
    param_plot = [dict(label=param_labels[i], values=param_values[i]) for i in range(len(param_values))]
    for d in param_plot:
        d['visible'] = True
    fig = go.Figure(data=
                   go.Parcoords(
                       line=dict(color=objective_values, 
                                 colorscale=colorscale,
                                 colorbar={'title': objective_label},
                                 reversescale=reverse_scale,
                                 showscale=True),
                       dimensions=param_plot
                   ))
    fig.update_layout(width=width, height=height)
    fig.show()


def plot(x, y, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    plt.plot(x, y, *args, **kwargs)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_ax(ax, x, y, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    ax.plot(x, y, *args, **kwargs)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def read_S21(file):
    df = pd.read_csv(f'{base}/{file}.csv')
    return np.array(df['Freq [GHz]']), np.array(df[df.keys()[1]])

def read_S11(file):
    df = pd.read_csv(f'{base}/{file}.csv')
    try:
        return np.array(df['Freq [GHz]']), np.array(df['dB(St(feedline1_T1,feedline1_T1)) []'])
    except KeyError:
        if 'dB(St(feedline_T1,feedline_T1)) []' in df:
            return np.array(df['Freq [GHz]']), np.array(df['dB(St(feedline_T1,feedline_T1)) []'])
        elif 'dB(St(Feedline_T1,Feedline_T1)) []' in df:
            return np.array(df['Freq [GHz]']), np.array(df['dB(St(Feedline_T1,Feedline_T1)) []'])
        else: 
            return read_S21(file) 


def S21_mag_old(f, f0, Qi, Qc):
    return abs(1- (get_Q(Qi, Qc)/Qc)/(1+2j*get_Q(Qi,Qc)*(f-f0)/f0))
    #return 1 - h*(w/2)/np.pi / ((w/2)**2 + (f-f0)**2)

def S21_mag(f, f0, Q, Qc, a, alpha, t, phi):
    if Q > Qc or Q < 0 or Qc < 0:
        return 1e6
    return abs( a*np.exp(1j*alpha-2*np.pi*1j*f*t)*(1- (Q/Qc)*np.exp(1j*phi)/(1+2j*Q*(f-f0)/f0)) )
    #return 1 - h*(w/2)/np.pi / ((w/2)**2 + (f-f0)**2)

def S11_mag(f, f0, Q, Qc, a, alpha, t, phi):
    if Q > Qc or Q < 0 or Qc < 0:
        return 1e6
    return abs( a*np.exp(1j*alpha-2*np.pi*1j*f*t)*(1- 2*(Q/Qc)*np.exp(1j*phi)/(1+2j*Q*(f-f0)/f0)) )

def S21_mag_noguchi(f, f0, kin, kex, a, phi):
    if kin < 0 or kex < 0: 
        return 1e6
    return abs( a*(1-kex*np.exp(1j*phi) / (1j*(f-f0) + (kin+kex)/2)))
    
def cutoff_x0(x, y, x0):
    if len(np.shape(x0)) > 0: 
        ind = np.where((x > x0[0]) & (x < x0[1]))
        return x[ind], y[ind]
    ind = np.where(x < x0)
    return x[ind], y[ind]

def dB_to_percent(dB):
    return 10**(dB/20)

def percent_to_dB(per):
    return 20*np.log10(per)

def get_S11_fit(f, S_dB, f0=2):
    init = [f0, 20, 30, 1, 0, 0, 0.5] 
    param = curve_fit(S11_mag, f, dB_to_percent(S_dB), p0=init, maxfev=100000)[0]
    return param

def get_S21_fit(f, S_dB, f0=2):
    init = [f0, 20, 30, 1, 0, 0, 0.5] 
    param = curve_fit(S21_mag, f, dB_to_percent(S_dB), p0=init, maxfev=100000)[0]
    return param

def get_S21_fit_noguchi(f, S_dB):
    init = [2, 2/30, 2/30, 1, 0.5]
    param = curve_fit(S21_mag_noguchi, f, dB_to_percent(S_dB), p0=init, maxfev=100000)[0]
    return param

def get_Qi(Q, Qc):
    return 1 / (1/Q - 1/Qc)

def get_Q(Qi, Qc):
    return 1 / (1/Qi + 1/Qc)

def plot_S21(name, cutoff_freq=3, f0_guess=2): 
    f, S = cutoff_x0(*read_S21(name), cutoff_freq)

    f0, Q, Qc_mag, a, alpha, t, phi = get_S21_fit(f, S, f0=f0_guess)
    plot(f, S, '.', label='Ansys Simulation')
    
    ft = np.linspace(min(f), max(f), 500000)
    St = S21_mag(ft, f0, Q, Qc_mag, a, alpha, t, phi)
    St_dB = percent_to_dB(St)

    Qc = Qc_mag*np.exp(-1j*phi).real
    plot(ft, St_dB, '--', label=f'Fit, $f_0$ = {f0:.2f}GHz, \n$Q_i$ = {get_Qi(Q, Qc):.2f}, \n$Q_c$ = {Qc:.2f}, \n$Q$ = {Q:.2f}')

    id0 = np.where(ft < f0)[0][-1]
    idl, idr = np.argmin(abs(St_dB[:id0]-(max(St_dB)-3))), id0+np.argmin(abs(St_dB[id0:]-(max(St_dB)-3)))
    plot(ft[idl], St_dB[idl], 'gX')
    plot(ft[idr], St_dB[idr], 'gX', label=f'-3dB, $Q$ = {f0/abs(ft[idr]-ft[idl]):.2f}', xlabel='f (GHz)', ylabel='$|S_{21}|$ (dB)')
    plt.legend()

def plot_S11(name, cutoff_freq=3, f0_guess=2): 
    f, S = cutoff_x0(*read_S11(name), cutoff_freq)

    f0, Q, Qc_mag, a, alpha, t, phi = get_S11_fit(f, S, f0=f0_guess)
    plot(f, S, '.', label='Ansys Simulation')
    
    ft = np.linspace(min(f), max(f), 500000)
    St = S11_mag(ft, f0, Q, Qc_mag, a, alpha, t, phi)
    St_dB = percent_to_dB(St)

    Qc = Qc_mag*np.exp(-1j*phi).real
    plot(ft, St_dB, '--', label=f'Fit, $f_0$ = {f0:.2f}GHz, \n$Q_i$ = {get_Qi(Q, Qc):.2f}, \n$Q_c$ = {Qc:.2f}, \n$Q$ = {Q:.2f}')

    id0 = np.where(ft < f0)[0][-1]
    idl, idr = np.argmin(abs(St_dB[:id0]-(max(St_dB)-3))), id0+np.argmin(abs(St_dB[id0:]-(max(St_dB)-3)))
    plot(ft[idl], St_dB[idl], 'gX')
    plot(ft[idr], St_dB[idr], 'gX', label=f'-3dB, $Q$ = {f0/abs(ft[idr]-ft[idl]):.2f}', xlabel='f (GHz)', ylabel='$|S_{11}|$ (dB)')
    plt.legend()
    #plt.show()


def remove_mm(data):
    data = np.array(data)
    data_rm = [i[:-2] for i in data]
    return np.array(data_rm, dtype=np.float64)