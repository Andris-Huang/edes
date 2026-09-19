import matplotlib.pyplot as plt
import numpy as np

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
    


def plot(x, y, *args, xlabel=None, ylabel=None, title=None, legend_outside = False, **kwargs):
    plt.plot(x, y, *args, **kwargs)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "label" in kwargs: 
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.legend()

def plot_ax(ax, x, y, *args, xlabel=None, ylabel=None, title=None, legend_outside=False, **kwargs):
    ax.plot(x, y, *args, **kwargs)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "label" in kwargs and kwargs["label"] != None: 
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
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

def generate_signal_timeline(commands):
    """
    Generates time (x) and value (y) arrays based on programmable commands.
    Supported commands:
      - ('flat', duration, value)
      - ('linear_sweep', duration, start_value, end_value)
    """
    x_data = []
    y_data = []
    current_time = 0.0
    
    for cmd_type, *args in commands:
        if cmd_type == 'flat':
            duration, value = args
            # Add start and end points for the flat segment
            x_data.extend([current_time, current_time + duration])
            y_data.extend([value, value])
            current_time += duration
            
        elif cmd_type == 'linear_sweep':
            duration, start_val, end_val = args
            # Generate a dense set of points for a smooth visual sweep (e.g., 100 points)
            sweep_times = np.linspace(current_time, current_time + duration, 100)
            sweep_vals = np.linspace(start_val, end_val, 100)
            
            x_data.extend(sweep_times.tolist())
            y_data.extend(sweep_vals.tolist())
            current_time += duration
            
    return x_data, y_data

def plot_generated_timelines(compiled_signals, title=None):
    num_signals = len(compiled_signals)
    fig, axes = plt.subplots(num_signals, 1, figsize=(12, 2.2 * num_signals), sharex=True)
    if num_signals == 1: axes = [axes]
    
    for i, (name, (x, y)) in enumerate(compiled_signals.items()):
        ax = axes[i]
        ax.plot(x, y, color=f'C{i}', linewidth=2)
        ax.fill_between(x, y, alpha=0.15, color=f'C{i}')
        
        # Dynamically scale Y axis based on data
        min_y, max_y = min(y), max(y)
        padding = max(1.0, (max_y - min_y) * 0.1)
        ax.set_ylim(min_y - padding, max_y + padding)
        
        ax.set_ylabel(name, rotation=0, labelpad=40, verticalalignment='center', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.xlabel("Time (seconds)")
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

