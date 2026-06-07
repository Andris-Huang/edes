import matplotlib.pyplot as plt

def big_plt_font():
    """
    plt.rcParams.update({'font.size': 14,
                         'lines.markersize': 12,
                         'lines.linewidth': 2.5,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'errorbar.capsize': 2})
    """
    plt.rcParams.update({'font.size': 14,
                         'lines.markersize': 12,
                         'lines.linewidth': 2.5,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'errorbar.capsize': 2})

def hollow_plt_font(): 
    plt.rcParams.update({'font.size': 14,
                         'lines.markersize': 9,
                         'lines.linewidth': 2.5,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'errorbar.capsize': 2, 
                         'lines.marker': 'o', 
                         'lines.markeredgewidth': 2,
                         'lines.markerfacecolor': 'none'})
    


def plot(x, y, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    plt.plot(x, y, *args, **kwargs)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "label" in kwargs: 
        plt.legend()

def plot_ax(ax, x, y, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    ax.plot(x, y, *args, **kwargs)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "label" in kwargs: 
        ax.legend()

def plot_errbar(x, y, yerr, xerr=None, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    plt.errorbar(x, y, yerr, xerr=xerr, *args, **kwargs)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def plot_ax_errbar(ax, x, y, yerr, xerr=None, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    ax.errorbar(x, y, yerr, xerr=xerr, *args, **kwargs)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_power_spectrum(freq, ideal_ps, real_ps):
    """
    Plots the power spectrum in dBm of an ideal signal
    and realistic signal side by side.
    """
    fig, ax = plt.subplots(ncols=2, figsize=(13, 5))
    plot_ax(ax[0], freq/1e6, ideal_ps, xlabel='Frequency (MHz)', ylabel='Power Spectrum (dBm)', title='Ideal Spectrum')
    plot_ax(ax[1], freq/1e6, real_ps, xlabel='Frequency (MHz)', title='Real Spectrum')

def plot_dot_dashed(x, y, *args, c=None, xlabel=None, ylabel=None, title=None, **kwargs):
    if c is None:
        p1 = plt.plot(x, y, 'o', *args, **kwargs)[0]
        plt.plot(x, y, color=p1.get_color(), alpha=0.4, *args, **kwargs)
    else: 
        plt.plot(x, y, 'o', *args, color=c, **kwargs)[0]
        plt.plot(x, y, color=c, alpha=0.4, linewidth=3, *args, **kwargs)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_ax_dot_dashed(ax, x, y, *args, c=None, xlabel=None, ylabel=None, title=None, **kwargs):
    if c is None:
        p1 = ax.plot(x, y, 'o', *args, **kwargs)[0]
        ax.plot(x, y, color=p1.get_color(), alpha=0.4, linewidth=3, *args, **kwargs)
    else: 
        ax.plot(x, y, 'o', *args, color=c, **kwargs)[0]
        ax.plot(x, y, color=c, alpha=0.4, linewidth=3, *args, **kwargs)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
 