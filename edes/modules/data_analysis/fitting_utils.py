from edes.modules.data_analysis import fitting_functions as fitting 
from edes.modules.data_analysis.spacetime_plot import plot_fitbounds, plot_fit
import numpy as np 
import matplotlib.pyplot as plt 

def plot_fitbounds(xdata,ydata,xfit,yfit,yT,yB,fit_params_dict,
                   xlabel=None, ylabel=None, xlim=None, fit_label='',
                   ylim=None, title=None, title_append=None, sigfigs=3, 
                   logx=False, logy=False):
    (fig, ax) = plt.subplots(figsize=(8, 5))

    fit_label = f'Fit {fit_label}\n'
    for key, value in fit_params_dict.items():
        fit_label += '{0} = {1:.{3}g} $\pm$ {2:.1g}\n'.format(key, value[0], value[1], sigfigs)
    fit_label = fit_label[:-1]
    
    ax.plot(xfit, yfit, color='k', label=fit_label)
    ax.plot(xdata, ydata, 'r.', markersize=12)
    ax.fill_between(xfit, yB, yT)
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=12)
    if logx: 
        ax.set_xscale('log') 
    if logy: 
        ax.set_yscale('log')
    plt.show()

def plot_fitbounds_ax(ax, xdata,ydata,xfit,yfit,yT,yB,fit_params_dict,*args,
                   xlabel=None, ylabel=None, xlim=None, fit_label='',
                   ylim=None, title=None, title_append=None, sigfigs=3, 
                   logx=False, logy=False, **kwargs):
    fit_label = f'{fit_label}\n'
    for key, value in fit_params_dict.items():
        fit_label += '{0} = {1:.{3}g} $\pm$ {2:.1g}\n'.format(key, value[0], value[1], sigfigs)
    fit_label = fit_label[:-1]
    
    line = ax.plot(xdata, ydata, '.', markersize=13, *args, **kwargs)
    ax.plot(xfit, yfit, '--', *args, alpha=0.8, c=line[0].get_color(), label=fit_label, **kwargs)
    ax.fill_between(xfit, yB, yT, *args, alpha=0.4, **kwargs)
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=12)
    if logx: 
        ax.set_xscale('log') 
    if logy: 
        ax.set_yscale('log')