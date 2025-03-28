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

def plot_errbar(x, y, yerr, xerr=None, *args, xlabel=None, ylabel=None, title=None, **kwargs):
    plt.plot(x, y, yerr, xerr=xerr, *args, **kwargs)
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