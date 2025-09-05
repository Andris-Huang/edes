import matplotlib.pyplot as plt
from datetime import date
import datetime

now = datetime.datetime.now()
date = str(date.today())
time = str(now.time())
xunits = ""
yunits = ""

def plot_data(x, y,
              xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, title_append=None):
    (fig, ax) = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, '.-', markersize=12)
    if xlabel is None:
        xlabel = xunits
    if ylabel is None:
        ylabel = yunits
    if title is None:
        title = '{0} - {1}'.format(date, time)
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    plt.show()

def plot_fitguess(x, ydata, yguess,
                  xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, title_append=None):
    (fig, ax) = plt.subplots(figsize=(8, 5))
    ax.plot(x, yguess, color='k', label='Fit guess')
    ax.plot(x, ydata, '.-', markersize=12, label='Data')
    if xlabel is None:
        xlabel = xunits
    if ylabel is None:
        ylabel = yunits
    if title is None:
        title = 'Fit Guess'
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    plt.show()
    

def plot_fit(xdata, ydata, xfit, yfit, fit_params_dict,
             xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, title_append=None, sigfigs=3):
    (fig, ax) = plt.subplots(figsize=(8, 5))
    
    fit_label = ''
    for key, value in fit_params_dict.items():
        fit_label += '{0} = {1:.{3}g} $\pm$ {2:.1g}\n'.format(key, value[0], value[1], sigfigs)
    fit_label = fit_label[:-1]
    
    ax.plot(xfit, yfit, color='k', label=fit_label)
    ax.plot(xdata, ydata, 'r.', markersize=12)
    if xlabel is None:
        xlabel = xunits
    if ylabel is None:
        ylabel = yunits
    if title is None:
        title = 'Fit: {0} - {1}'.format(date, time)
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.show()
    

def plot_fitbounds(xdata,ydata,xfit,yfit,yT,yB,fit_params_dict,
                  xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, title_append=None, sigfigs=3):
    (fig, ax) = plt.subplots(figsize=(8, 5))
    
    fit_label = ''
    for key, value in fit_params_dict.items():
        fit_label += '{0} = {1:.{3}g} $\pm$ {2:.1g}\n'.format(key, value[0], value[1], sigfigs)
    fit_label = fit_label[:-1]
    
    ax.plot(xfit, yfit, color='k', label=fit_label)
    ax.plot(xdata, ydata, 'r.', markersize=12)
    ax.fill_between(xfit, yB, yT)
    if xlabel is None:
        xlabel = xunits
    if ylabel is None:
        ylabel = yunits
    if title is None:
        title = 'Fit: {0} - {1}'.format(date, time)
    if title_append is not None:
        title += title_append
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.show()