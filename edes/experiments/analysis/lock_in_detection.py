import edes 
from edes.utils.plotting import plot, plot_ax, big_plt_font, plot_ax_errbar, plot_errbar
from edes.utils.plotting import plot_generated_timelines, generate_signal_timeline
from edes.utils.circuits import dBm_to_W, W_to_dBm
from edes.utils.utils import U2_to_MHz
from edes.utils.file_handling import load_h5_data, load_latest_data, \
                                     load_latest_filename, list_filenames, convert_dict_to_dataframe, \
                                     find_file_by_rid, load_artiq_h5_arguments, get_artiq_event_files
import numpy as np 
from tqdm import tqdm, trange 
import matplotlib.pyplot as plt 
import time
import pandas as pd
import re
import os
import glob
import ipywidgets as widgets
from IPython.display import display, clear_output
big_plt_font()


def plot_hist_across_var(tip_on_files, tip_off_files, data_dir, var='t_load', fix_xlim=False):
    """
    Plots histograms of detected power for Tip-On and Tip-Off datasets across a specified variable.

    Parameters:
    - tip_on_files: List of file paths for Tip-On datasets.
    - tip_off_files: List of file paths for Tip-Off datasets.
    - data_dir: Base directory where the data files are located.
    - var: The variable to use for mapping datasets (default is 't_load').
    - fix_xlim: Boolean (default False). If True, the x-axis limits initialize as locked 
                to the global min and max power values across all scanned datasets.
    """

    on_map = {}
    off_map = {}

    # We will track the absolute minimum and maximum power values in nW across all datasets
    global_x_min = float('inf')
    global_x_max = float('-inf')

    print(f"Scanning datasets for {var} and calculating global x-limits...")
    
    # Scan Tip-On files
    for f in tip_on_files:
        try:
            f_data = load_h5_data(f, base=data_dir)
            t_val = f_data.get(f'{var}', None)
            if t_val is not None:
                on_map[t_val] = f
                
            # Convert values to nW to find global limits
            data = f_data.get('all_meas', [])
            if len(data) > 0:
                p_nw_vals = dBm_to_W(np.concatenate(data, axis=0)) * 1e9
                global_x_min = min(global_x_min, np.min(p_nw_vals))
                global_x_max = max(global_x_max, np.max(p_nw_vals))
        except Exception as e:
            print(f"Skipping Tip-On file {f}: {e}")

    # Scan Tip-Off files
    for f in tip_off_files:
        try:
            f_data = load_h5_data(f, base=data_dir)
            t_val = f_data.get(f'{var}', None)
            if t_val is not None:
                off_map[t_val] = f
                
            # Convert values to nW to find global limits
            data2 = f_data.get('all_meas', [])
            if len(data2) > 0:
                p_nw_vals_bg = dBm_to_W(np.concatenate(data2, axis=0)) * 1e9
                global_x_min = min(global_x_min, np.min(p_nw_vals_bg))
                global_x_max = max(global_x_max, np.max(p_nw_vals_bg))
        except Exception as e:
            print(f"Skipping Tip-Off file {f}: {e}")

    unique_t_loads = sorted(list(on_map.keys()))
    print(f"Scan complete. Found {len(unique_t_loads)} valid {var} configurations.")

    # Calculate padding for fixed limits
    has_global_limits = (global_x_min != float('inf') and global_x_max != float('-inf'))
    if has_global_limits:
        x_range = global_x_max - global_x_min
        padding = x_range * 0.05 if x_range > 0 else 1.0
        global_x_min -= padding
        global_x_max += padding
    else:
        global_x_min, global_x_max = 0, 10 # Fallbacks

    # 2. Setup UI elements (Dropdown, Buttons, Checkbox)
    t_load_dropdown = widgets.Dropdown(
        options=unique_t_loads,
        value=unique_t_loads[0] if unique_t_loads else None,
        description=f'{var}:'
    )

    btn_prev = widgets.Button(
        description='Prev',
        button_style='info',     
        icon='arrow-left'        
    )

    btn_next = widgets.Button(
        description='Next',
        button_style='primary',  
        icon='arrow-right'       
    )

    fix_xlim_checkbox = widgets.Checkbox(
        value=fix_xlim,
        description='Fix X-Axis Limits',
        disabled=not has_global_limits,
        indent=False
    )

    plot_output = widgets.Output()

    # 3. Plotting logic wrapped in an output handler
    def draw_plot(change=None):
        t_load = t_load_dropdown.value
        
        with plot_output:
            clear_output(wait=True)
            
            if t_load is None:
                print(f"No {var} configured.")
                return

            on_file_path = on_map.get(t_load)
            off_file_path = off_map.get(t_load) or (list(off_map.values())[0] if off_map else None)

            if not on_file_path:
                print(f"No Tip-On file found for {var}: {t_load}")
                return

            # Load file data
            file = load_h5_data(on_file_path, base=data_dir)
            data = file['all_meas']
            
            # Calculate Power in nW for Tip-On
            P_nW = dBm_to_W(np.concatenate(data, axis=0)) * 1e9 
            bin_size = 0.01

            # Plotting Setup
            fig, ax = plt.subplots(figsize=(7, 4))

            # 1. Plot Tip On Histogram
            counts, bin_edges = np.histogram(P_nW, bins=int((max(P_nW) - min(P_nW)) / bin_size))
            ax.bar(
                bin_edges[:-1], 
                counts / np.sum(counts), 
                width=np.diff(bin_edges),
                label='Tip on',
                align='edge', 
                alpha=0.6,
                color='blue'
            )

            # 2. Plot Tip Off Histogram (if background file exists)
            if off_file_path:
                file2 = load_h5_data(off_file_path, base=data_dir)
                data2 = file2['all_meas']
                P_nW_background = dBm_to_W(np.concatenate(data2, axis=0)) * 1e9

                counts_bg, bin_edges_bg = np.histogram(
                    P_nW_background, 
                    bins=int((max(P_nW_background) - min(P_nW_background)) / bin_size)
                )
                ax.bar(
                    bin_edges_bg[:-1], 
                    counts_bg / np.sum(counts_bg), 
                    width=np.diff(bin_edges_bg),
                    label='Tip off',
                    align='edge', 
                    alpha=0.6,
                    color='orange'
                )

            # Apply locked limits if requested
            if fix_xlim_checkbox.value:
                ax.set_xlim(global_x_min, global_x_max)

            # Graph configurations
            ax.set_yscale('log')
            ax.set_title(f"Histogram of Detected Power | {var} = {t_load}")
            ax.set_xlabel('Detected Power (nW)') 
            ax.set_ylabel('Probability Density / 0.01nW bin')
            ax.grid(True)
            ax.legend()
            plt.show()

    # 4. Define event behaviors for Slider/Buttons
    def on_dropdown_change(change):
        draw_plot(change['new'])

    def on_next_clicked(b):
        opts = t_load_dropdown.options
        if not opts: return
        idx = opts.index(t_load_dropdown.value)
        t_load_dropdown.value = opts[idx + 1] if idx < len(opts) - 1 else opts[0]

    def on_prev_clicked(b):
        opts = t_load_dropdown.options
        if not opts: return
        idx = opts.index(t_load_dropdown.value)
        t_load_dropdown.value = opts[idx - 1] if idx > 0 else opts[-1]

    # Link actions to the widgets
    t_load_dropdown.observe(on_dropdown_change, names='value')
    # Automatically redraw when the user switches fixed limits
    fix_xlim_checkbox.observe(lambda change: draw_plot(), names='value')
    
    btn_next.on_click(on_next_clicked)
    btn_prev.on_click(on_prev_clicked)

    # 5. Pack layouts and display
    controls = widgets.HBox([btn_prev, t_load_dropdown, btn_next, fix_xlim_checkbox])
    ui = widgets.VBox([controls, plot_output])

    display(ui)

    # Render the first plot
    if t_load_dropdown.value is not None:
        draw_plot(t_load_dropdown.value)

def plot_PvT_across_var(tip_on_files, data_dir, var='t_load', fix_ylim=False):
    """
    Plots interactive individual Power vs. Time (PvT) sweeps across a specified variable.

    Parameters:
    - tip_on_files: List of file paths for Tip-On datasets.
    - data_dir: Base directory where the data files are located.
    - var: The variable to use for mapping datasets (default is 't_load').
    - fix_ylim: Boolean (default False). If True, the y-axis limits initialize as locked 
                to the global min and max across all datasets.
    """

    on_map = {}
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    print(f"Scanning datasets for {var} and calculating global y-limits...")
    for f in tip_on_files:
        try:
            f_data = load_h5_data(f, base=data_dir)
            t_val = f_data.get(f'{var}', None)
            if t_val is not None:
                on_map[t_val] = f
                
            # Scan the arrays in this file to update global y-limits
            data_arrays = f_data.get('all_meas', [])
            if len(data_arrays) > 0:
                file_min = np.min(data_arrays)
                file_max = np.max(data_arrays)
                if file_min < global_y_min:
                    global_y_min = file_min
                if file_max > global_y_max:
                    global_y_max = file_max
                    
        except Exception as e:
            print(f"Skipping Tip-On file {f}: {e}")

    unique_t_loads = sorted(list(on_map.keys()))
    print(f"Scan complete. Found {len(unique_t_loads)} valid {var} configurations.")
    
    # Check if we successfully obtained valid limits
    has_global_limits = (global_y_min != float('inf') and global_y_max != float('-inf'))
    if has_global_limits:
        # Add a tiny bit of padding (e.g., 5%) so curves don't hug the very edge of the plot
        y_range = global_y_max - global_y_min
        padding = y_range * 0.05 if y_range > 0 else 1.0
        global_y_min -= padding
        global_y_max += padding
    else:
        global_y_min, global_y_max = -100, 10  # Fallbacks

    if not unique_t_loads:
        print("Error: No valid datasets matched the specified variable.")
        return

    # 2. Setup UI elements
    t_load_dropdown = widgets.Dropdown(
        options=unique_t_loads,
        value=unique_t_loads[0],
        description=f'{var}:'
    )

    btn_var_prev = widgets.Button(
        description='Prev Config',
        button_style='info',
        icon='arrow-left'
    )

    btn_var_next = widgets.Button(
        description='Next Config',
        button_style='primary',
        icon='arrow-right'
    )

    # Individual dataset index slider
    idx_slider = widgets.IntSlider(
        value=0, 
        min=0, 
        max=0, 
        step=1, 
        description='Index (i):'
    )

    btn_idx_prev = widgets.Button(
        description='Prev Sweep',
        button_style='info',
        icon='arrow-left'
    )

    btn_idx_next = widgets.Button(
        description='Next Sweep',
        button_style='primary',
        icon='arrow-right'
    )

    # Checkbox initialized with the function's parameter value
    fix_ylim_checkbox = widgets.Checkbox(
        value=fix_ylim,
        description='Fix Y-Axis Limits',
        disabled=not has_global_limits,
        indent=False
    )

    plot_output = widgets.Output()

    # 3. Plotting Logic
    def draw_plot(change=None):
        t_load = t_load_dropdown.value
        i = idx_slider.value

        with plot_output:
            clear_output(wait=True)
            
            if t_load is None:
                print(f"No {var} configured.")
                return

            on_file_path = on_map.get(t_load)
            if not on_file_path:
                print(f"No Tip-On file found for {var}: {t_load}")
                return

            # Load file data
            file = load_h5_data(on_file_path, base=data_dir)
            data = file['all_meas'] 
            t_seq_end = file['t_data']
            t_axis = np.linspace(0, t_seq_end, len(data[0])) 
            
            # Bound check
            if i >= len(data):
                i = len(data) - 1

            # Plotting Setup
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_axis * 1e3, data[i], color='blue', lw=2)
            
            # Apply fixed limits if the checkbox is active
            if fix_ylim_checkbox.value:
                ax.set_ylim(global_y_min, global_y_max)
            
            ax.set_title(f"Plotting {var} = {t_load} | Index i = {i}")
            ax.set_xlabel("Wait time (ms)")
            ax.set_ylabel("Power (dBm)")
            ax.grid(True)
            plt.show()

    # 4. Dynamic Options & Slider Bound Updater
    def update_file_selection(*args):
        t_load = t_load_dropdown.value
        on_file_path = on_map.get(t_load)
        
        if on_file_path:
            file = load_h5_data(on_file_path, base=data_dir)
            data = file['all_meas']
            
            idx_slider.unobserve(draw_plot, names='value')
            idx_slider.max = len(data) - 1
            idx_slider.value = 0
            idx_slider.observe(draw_plot, names='value')
            
            draw_plot()
        else:
            with plot_output:
                clear_output(wait=True)
                print(f"No file path mapping found for {t_load}")

    # 5. Define Button Actions
    def on_var_prev_clicked(b):
        opts = t_load_dropdown.options
        if not opts: return
        idx = opts.index(t_load_dropdown.value)
        t_load_dropdown.value = opts[idx - 1] if idx > 0 else opts[-1]

    def on_var_next_clicked(b):
        opts = t_load_dropdown.options
        if not opts: return
        idx = opts.index(t_load_dropdown.value)
        t_load_dropdown.value = opts[idx + 1] if idx < len(opts) - 1 else opts[0]

    def on_idx_prev_clicked(b):
        if idx_slider.value > idx_slider.min:
            idx_slider.value -= 1
        else:
            idx_slider.value = idx_slider.max

    def on_idx_next_clicked(b):
        if idx_slider.value < idx_slider.max:
            idx_slider.value += 1
        else:
            idx_slider.value = idx_slider.min

    # 6. Link Observers and On-Click Events
    t_load_dropdown.observe(update_file_selection, names='value')
    idx_slider.observe(draw_plot, names='value')
    fix_ylim_checkbox.observe(draw_plot, names='value')

    btn_var_prev.on_click(on_var_prev_clicked)
    btn_var_next.on_click(on_var_next_clicked)
    btn_idx_prev.on_click(on_idx_prev_clicked)
    btn_idx_next.on_click(on_idx_next_clicked)

    # Initialize first run
    update_file_selection()

    # 7. Package and Display Layouts
    var_controls = widgets.HBox([btn_var_prev, t_load_dropdown, btn_var_next])
    sweep_controls = widgets.HBox([btn_idx_prev, idx_slider, btn_idx_next, fix_ylim_checkbox])
    
    ui = widgets.VBox([
        var_controls, 
        sweep_controls, 
        plot_output
    ])

    display(ui)

def plot_hist(file=None, background_file=None, data_dir=''): 
    if file is None:
        file = load_latest_data()#load_h5_data('/home/electron/data/experiment_07142026/RFPowerSweepOnOffDifferential_11-55-06.h5') #
    else: 
        file = load_h5_data(file, base=data_dir)
    data = file['all_meas'] 
    if background_file is None:
        file2 = load_latest_data()
    else:
        file2 = load_h5_data(background_file, base=data_dir)
    data2 = file2['all_meas'] 
    P_nW = dBm_to_W(np.concatenate(data, axis=0))*1e9 
    P_nW_background = dBm_to_W(np.concatenate(data2, axis=0))*1e9
    bin_size = 0.01

    counts, bin_edges = np.histogram(P_nW, bins=int((max(P_nW)-min(P_nW))/bin_size))#, alpha=0.7, label='Tip on')
    plt.bar(
        bin_edges[:-1],          # X-coordinates: Use all edges except the very last one
        counts/np.sum(counts),                  # Y-coordinates: The heights of the bars
        width=np.diff(bin_edges),# Width of each bar (difference between adjacent edges)
        label='Tip on',
        align='edge',            # Align the bars to start at the left bin edge
        alpha=0.6                # Transparency
    )

    counts, bin_edges = np.histogram(P_nW_background, bins=int((max(P_nW_background)-min(P_nW_background))/bin_size))#, alpha=0.7, label='Tip off')
    # plt.xscale('log') 
    plt.bar(
        bin_edges[:-1],          # X-coordinates: Use all edges except the very last one
        counts/np.sum(counts),                  # Y-coordinates: The heights of the bars
        width=np.diff(bin_edges),# Width of each bar (difference between adjacent edges)
        label='Tip off',
        align='edge',            # Align the bars to start at the left bin edge
        alpha=0.6                # Transparency
    )



    plt.yscale('log')
    plt.xlabel('Detected Power (nW)') 
    plt.ylabel('Probability Density / 0.1nW bin')
    plt.grid()
    plt.legend()
    plt.show()

def plot_individual_PvT(file=None, data_dir='', fix_ylim=False): 
    
    # --- Data loading setup ---
    if file is None:
        file = load_latest_data() 
    else:
        file = load_h5_data(file, base=data_dir)
        
    data = file['all_meas'] 
    t_seq_end = file['t_data']
    x_data = t_seq_end * 1e3
    t_axis = np.linspace(0, x_data, len(data[0])) 

    # Calculate limits across all datasets in this single file
    file_min = np.min(data)
    file_max = np.max(data)
    
    # Add 5% padding so curves don't clip at the edges
    y_range = file_max - file_min
    padding = y_range * 0.05 if y_range > 0 else 1.0
    global_y_min = file_min - padding
    global_y_max = file_max + padding

    # 1. Setup UI elements
    slider = widgets.IntSlider(
        value=0, 
        min=0, 
        max=len(data) - 1, 
        step=1, 
        description='Index (i):'
    )

    btn_prev = widgets.Button(
        description='Prev',
        button_style='info',     
        icon='arrow-left'        
    )

    btn_next = widgets.Button(
        description='Next',
        button_style='primary',  
        icon='arrow-right'       
    )

    fix_ylim_checkbox = widgets.Checkbox(
        value=fix_ylim,
        description='Fix Y Limits',
        indent=False
    )

    # Custom x-limit UI elements
    custom_xlim_checkbox = widgets.Checkbox(
        value=False,
        description='Enable Custom X Limits',
        indent=False
    )

    xlim_min_input = widgets.FloatText(
        value=0.0,
        description='X Min:',
        layout=widgets.Layout(width='150px')
    )

    xlim_max_input = widgets.FloatText(
        value=float(np.max(t_axis)),
        description='X Max:',
        layout=widgets.Layout(width='150px')
    )

    # Output container to catch the plot and refresh it cleanly
    plot_output = widgets.Output()

    # 2. Plotting logic wrapped in an output handler
    def draw_plot(i=None):
        index = slider.value if i is None or isinstance(i, dict) else i
        
        with plot_output:
            clear_output(wait=True)  # Instantly clears the old plot/canvas
            
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_axis, data[index], color='blue', lw=2)
            
            # Apply fixed y-limits if requested
            if fix_ylim_checkbox.value:
                ax.set_ylim(global_y_min, global_y_max)
            
            # Apply custom user-defined x-limits if enabled
            if custom_xlim_checkbox.value:
                ax.set_xlim(xlim_min_input.value, xlim_max_input.value)
            
            ax.set_title(f"Plotting dataset i = {index}")
            ax.set_xlabel("Wait time (ms)")
            ax.set_ylabel("Power (dBm)")
            ax.grid(True)
            plt.show()

    # 3. Define event behaviors
    def on_slider_change(change):
        draw_plot(change['new'])

    def on_next_clicked(b):
        if slider.value < slider.max:
            slider.value += 1
        else:
            slider.value = 0

    def on_prev_clicked(b):
        if slider.value > slider.min:
            slider.value -= 1
        else:
            slider.value = slider.max

    # Trigger redraw when inputs or checkboxes are modified
    def trigger_redraw(change):
        draw_plot()

    # Link actions to the widgets
    slider.observe(on_slider_change, names='value')
    fix_ylim_checkbox.observe(trigger_redraw, names='value')
    custom_xlim_checkbox.observe(trigger_redraw, names='value')
    xlim_min_input.observe(trigger_redraw, names='value')
    xlim_max_input.observe(trigger_redraw, names='value')
    
    btn_next.on_click(on_next_clicked)
    btn_prev.on_click(on_prev_clicked)

    # 4. Pack layouts into clean, grouped sections and display
    controls = widgets.HBox([btn_prev, slider, btn_next, fix_ylim_checkbox])
    x_limit_controls = widgets.HBox([custom_xlim_checkbox, xlim_min_input, xlim_max_input])
    
    ui = widgets.VBox([controls, x_limit_controls, plot_output])

    display(ui)

    # Render the initial plot
    draw_plot(slider.value)

def compare_PvT_U2_Ex_continuous_loading(data_files, data_dir='', fix_ylim=False):
    """
    Plots interactive PvT curves across Ex, U2, and RF Power configurations.

    Parameters:
    - data_files: List of file paths to process.
    - data_dir: Base directory for the data files.
    - fix_ylim: Boolean (default False). If True, the y-axis limits initialize as locked 
                to the global min and max signal values across all scanned configurations.
    """

    # 1. Gather all files, map configurations, and calculate global y-limits
    h5_files = data_files 
    experiment_map = {}
    
    global_y_min = float('inf')
    global_y_max = float('-inf')

    print("Scanning datasets and calculating global y-limits...")
    for filepath in h5_files:
        try:
            data_dict = load_h5_data(filepath, base=data_dir)
            ex_val = data_dict.get('Ex', None)
            u2_val = data_dict.get('U2', None)
            
            if ex_val is not None and u2_val is not None:
                experiment_map[(ex_val, u2_val)] = filepath
                
                # Load dataframe to find global min/max limits of y-data ('data')
                df = convert_dict_to_dataframe(data_dict)
                for raw_data in df['data'].values:
                    y_arr = np.array(raw_data)
                    if len(y_arr) > 0:
                        global_y_min = min(global_y_min, np.min(y_arr))
                        global_y_max = max(global_y_max, np.max(y_arr))
        except Exception as e:
            print(f"Skipping {filepath} due to error: {e}")

    unique_Ex = sorted(list(set(k[0] for k in experiment_map.keys())))
    unique_U2 = sorted(list(set(k[1] for k in experiment_map.keys())))
    print(f"Scan complete. Found {len(experiment_map)} valid Ex/U2 configurations.")

    # Apply global limits calculations with padding buffer
    has_global_limits = (global_y_min != float('inf') and global_y_max != float('-inf'))
    if has_global_limits:
        y_range = global_y_max - global_y_min
        padding = y_range * 0.05 if y_range > 0 else 1.0
        global_y_min -= padding
        global_y_max += padding
    else:
        global_y_min, global_y_max = -100, 10  # Fallbacks

    # 2. Set up the UI Elements
    ex_dropdown = widgets.Dropdown(options=unique_Ex, description='Ex:')
    u2_dropdown = widgets.Dropdown(options=unique_U2, description='U2:')
    power_dropdown = widgets.Dropdown(description='Power (dBm):')

    btn_prev = widgets.Button(description='Prev', button_style='info', icon='arrow-left')
    btn_next = widgets.Button(description='Next', button_style='primary', icon='arrow-right')

    fix_ylim_checkbox = widgets.Checkbox(
        value=fix_ylim,
        description='Fix Y-Axis Limits',
        disabled=not has_global_limits,
        indent=False
    )

    plot_output = widgets.Output()

    # 4. Plotting logic wrapped in the output handler
    def draw_plot(change=None):
        Ex = ex_dropdown.value
        U2 = u2_dropdown.value
        Power_dBm = power_dropdown.value

        with plot_output:
            clear_output(wait=True)
            if Ex is None or U2 is None or Power_dBm is None:
                print("Awaiting valid selection...")
                return
                
            filepath = experiment_map.get((Ex, U2))
            if not filepath:
                print(f"No file found for Ex: {Ex}, U2: {U2}")
                return
                
            # Load dataset and convert
            data_dict = load_h5_data(filepath, base=data_dir)
            df = convert_dict_to_dataframe(data_dict)
            
            # Find the row for the selected power
            dbm_series = df['dBm'].astype(float)
            row = df[np.isclose(dbm_series, float(Power_dBm), atol=1e-3)]
            
            if row.empty:
                print(f"Power {Power_dBm} dBm not found in this dataset.")
                return
                
            # Extract time-domain data for the specific power
            raw_data = row['data'].values[0]
            y_data = np.array(raw_data)
            
            # Extract t_data for the x-axis
            t_axis = np.array(row['time'].values[0])
            
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_axis, y_data, color='blue', lw=2)
            
            # Apply locked limits if requested
            if fix_ylim_checkbox.value:
                ax.set_ylim(global_y_min, global_y_max)
            
            ax.set_title(f"Ex: {Ex} | U2: {U2} | RF Power: {Power_dBm} dBm")
            ax.set_xlabel("Wait time (s)")
            ax.set_ylabel("Power (dBm)")
            ax.grid(True)
            plt.show()

    # 3. Dynamic options updater for the Power dropdown
    def update_power_options(*args):
        filepath = experiment_map.get((ex_dropdown.value, u2_dropdown.value))
        if filepath:
            # 1. Store the currently selected power value
            current_power = power_dropdown.value

            # Load available powers for the selected Ex and U2
            data_dict = load_h5_data(filepath, base=data_dir)
            df = convert_dict_to_dataframe(data_dict)
            new_options = sorted(df['dBm'].unique())
            
            # Temporarily unobserve to avoid triggering draw_plot during options/value setup
            power_dropdown.unobserve(draw_plot, names='value')
            
            power_dropdown.options = new_options
            
            if new_options:
                # 2. Check if the previously selected power exists in the new dataset
                matching_option = next(
                    (opt for opt in new_options if np.isclose(opt, current_power, atol=1e-3)), 
                    None
                ) if current_power is not None else None
                
                # Keep current power if available, otherwise default to first entry
                if matching_option is not None:
                    power_dropdown.value = matching_option
                else:
                    power_dropdown.value = new_options[0]
                
            # Re-observe now that values are stabilized
            power_dropdown.observe(draw_plot, names='value')
            
            # Manually trigger the redraw for the newly selected file/power combination
            draw_plot()
        else:
            power_dropdown.options = []
            draw_plot()

    # 5. Define Button Behaviors (Step through the available powers)
    def on_prev_clicked(b):
        opts = power_dropdown.options
        if not opts: return
        idx = opts.index(power_dropdown.value)
        power_dropdown.value = opts[idx - 1] if idx > 0 else opts[-1]

    def on_next_clicked(b):
        opts = power_dropdown.options
        if not opts: return
        idx = opts.index(power_dropdown.value)
        power_dropdown.value = opts[idx + 1] if idx < len(opts) - 1 else opts[0]

    # 6. Link Observers and Clicks
    ex_dropdown.observe(update_power_options, names='value')
    u2_dropdown.observe(update_power_options, names='value')
    power_dropdown.observe(draw_plot, names='value')
    fix_ylim_checkbox.observe(lambda change: draw_plot(), names='value')

    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)

    # Initialize the first state
    update_power_options()

    # 7. Layout and Display
    param_controls = widgets.HBox([ex_dropdown, u2_dropdown])
    power_controls = widgets.HBox([btn_prev, power_dropdown, btn_next, fix_ylim_checkbox])

    ui = widgets.VBox([
        param_controls,
        power_controls, 
        plot_output
    ])

    display(ui)

def compare_hist_U2_Ex_continuous_loading(data_files, data_dir='', fix_xlim=False):
    """
    Plots interactive power histograms across Ex, U2, and RF Power configurations.

    Parameters:
    - data_files: List of file paths to process.
    - data_dir: Base directory for the data files.
    - fix_xlim: Boolean (default False). If True, the x-axis limits initialize as locked 
                to the global min and max power values across all scanned configurations.
    """

    # 1. Gather all files, map configurations, and calculate global x-limits
    h5_files = data_files 
    experiment_map = {}
    
    global_x_min = float('inf')
    global_x_max = float('-inf')

    print("Scanning datasets and calculating global x-limits...")
    for filepath in h5_files:
        try:
            data_dict = load_h5_data(filepath, base=data_dir)
            ex_val = data_dict.get('Ex', None)
            u2_val = data_dict.get('U2', None)
            
            if ex_val is not None and u2_val is not None:
                experiment_map[(ex_val, u2_val)] = filepath
                
                # Load dataframe to extract raw data and find global min/max limits
                df = convert_dict_to_dataframe(data_dict)
                for raw_data in df['data'].values:
                    data_nw = dBm_to_W(np.array(raw_data)) * 1e9
                    if len(data_nw) > 0:
                        global_x_min = min(global_x_min, np.min(data_nw))
                        global_x_max = max(global_x_max, np.max(data_nw))
        except Exception as e:
            print(f"Skipping {filepath} due to error: {e}")

    unique_Ex = sorted(list(set(k[0] for k in experiment_map.keys())))
    unique_U2 = sorted(list(set(k[1] for k in experiment_map.keys())))
    print(f"Scan complete. Found {len(experiment_map)} valid Ex/U2 configurations.")

    # Apply global x-limits calculation with a 5% padding buffer
    has_global_limits = (global_x_min != float('inf') and global_x_max != float('-inf'))
    if has_global_limits:
        x_range = global_x_max - global_x_min
        padding = x_range * 0.05 if x_range > 0 else 1.0
        global_x_min -= padding
        global_x_max += padding
    else:
        global_x_min, global_x_max = 0, 10  # Fallbacks

    # 2. Set up the UI Elements
    ex_dropdown = widgets.Dropdown(options=unique_Ex, description='Ex:')
    u2_dropdown = widgets.Dropdown(options=unique_U2, description='U2:')
    power_dropdown = widgets.Dropdown(description='Power (dBm):')

    btn_prev = widgets.Button(description='Prev', button_style='info', icon='arrow-left')
    btn_next = widgets.Button(description='Next', button_style='primary', icon='arrow-right')

    fix_xlim_checkbox = widgets.Checkbox(
        value=fix_xlim,
        description='Fix X-Axis Limits',
        disabled=not has_global_limits,
        indent=False
    )

    plot_output = widgets.Output()

    # 4. Plotting logic wrapped in the output handler
    def draw_plot(change=None):
        Ex = ex_dropdown.value
        U2 = u2_dropdown.value
        Power_dBm = power_dropdown.value

        with plot_output:
            clear_output(wait=True)
            if Ex is None or U2 is None or Power_dBm is None:
                print("Awaiting valid selection...")
                return
                
            filepath = experiment_map.get((Ex, U2))
            if not filepath:
                print(f"No file found for Ex: {Ex}, U2: {U2}")
                return
                
            # Load dataset and convert
            data_dict = load_h5_data(filepath, base=data_dir)
            df = convert_dict_to_dataframe(data_dict)
            
            # Find the row for the selected power
            dbm_series = df['dBm'].astype(float)
            row = df[np.isclose(dbm_series, float(Power_dBm), atol=1e-3)]
            
            if row.empty:
                print(f"Power {Power_dBm} dBm not found in this dataset.")
                return
                
            # Extract time-domain data for the specific power
            raw_data = row['data'].values[0]
            data = dBm_to_W(np.array(raw_data)) * 1e9
            
            # Calculate metric (fraction of data > 10x mean)
            mean_val = np.mean(data)
            if mean_val > 0:
                max_data = len(data[data > 10 * mean_val]) / len(data)
            else:
                max_data = 0
            
            # Plotting
            plt.figure(figsize=(8, 5))
            plt.hist(data, density=True, bins=50, alpha=0.75, color='b', edgecolor='black')
            plt.title(f'Ex: {Ex} | U2: {U2} | Power: {Power_dBm:.2f} dBm\nMetric ( >10x Mean Ratio): {max_data:.4f}')
            
            # Apply locked limits if requested
            if fix_xlim_checkbox.value:
                plt.xlim(global_x_min, global_x_max)
                
            plt.xlabel('Power (nW)')
            plt.ylabel('Density (Log Scale)')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.show()

    # 3. Dynamic options updater for the Power dropdown
    def update_power_options(*args):
        filepath = experiment_map.get((ex_dropdown.value, u2_dropdown.value))
        if filepath:
            # 1. Temporarily store the currently selected power value
            current_power = power_dropdown.value

            # Load available powers for the selected Ex and U2
            data_dict = load_h5_data(filepath, base=data_dir)
            df = convert_dict_to_dataframe(data_dict)
            new_options = sorted(df['dBm'].unique())
            
            # Temporarily unobserve to avoid triggering draw_plot during options/value setup
            power_dropdown.unobserve(draw_plot, names='value')
            
            power_dropdown.options = new_options
            
            if new_options:
                # 2. Check if current_power is compatible with any value in the new options list
                matching_option = next(
                    (opt for opt in new_options if np.isclose(opt, current_power, atol=1e-3)), 
                    None
                ) if current_power is not None else None
                
                # Keep the previous selection if it is available, otherwise default to first available
                if matching_option is not None:
                    power_dropdown.value = matching_option
                else:
                    power_dropdown.value = new_options[0]
                
            # Re-observe now that values are stabilized
            power_dropdown.observe(draw_plot, names='value')
            
            # Manually trigger the redraw for the newly selected file/power combination
            draw_plot()
        else:
            power_dropdown.options = []
            draw_plot()

    # 5. Define Button Behaviors (Step through the available powers)
    def on_prev_clicked(b):
        opts = power_dropdown.options
        if not opts: return
        idx = opts.index(power_dropdown.value)
        power_dropdown.value = opts[idx - 1] if idx > 0 else opts[-1]

    def on_next_clicked(b):
        opts = power_dropdown.options
        if not opts: return
        idx = opts.index(power_dropdown.value)
        power_dropdown.value = opts[idx + 1] if idx < len(opts) - 1 else opts[0]

    # 6. Link Observers and Clicks
    ex_dropdown.observe(update_power_options, names='value')
    u2_dropdown.observe(update_power_options, names='value')
    power_dropdown.observe(draw_plot, names='value')
    fix_xlim_checkbox.observe(lambda change: draw_plot(), names='value')

    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)

    # Initialize the first state
    update_power_options()

    # 7. Layout and Display
    param_controls = widgets.HBox([ex_dropdown, u2_dropdown])
    power_controls = widgets.HBox([btn_prev, power_dropdown, btn_next, fix_xlim_checkbox])

    ui = widgets.VBox([
        param_controls,
        power_controls, 
        plot_output
    ])

    display(ui)

def plot_Prob_vs_var(tip_on_files, tip_off_files, data_dir, var='t_load', 
                     manual_var_vales=None,
                     sigma_cutoff=8,xlabel='Loading time (s)'):
    """
    Plots the probability of detected power exceeding a threshold for Tip-On and Tip-Off datasets across a specified variable.

    Parameters:
    - tip_on_files: List of file paths for Tip-On datasets.
    - tip_off_files: List of file paths for Tip-Off datasets.
    - data_dir: Base directory where the data files are located.
    - var: The variable to use for mapping datasets (default is 't_load').
    """
    # --- Helper functions (Ensure dBm_to_W, load_h5_data, and data_dir are defined) ---
    # def dBm_to_W(dbm): ...
    # def load_h5_data(filepath, base=None): ...
    # data_dir = '/home/electron/data/experiment_07142026'

    file2 = load_h5_data(tip_off_files[0], base=data_dir) 
    data2 = file2['all_meas'] 
    P_nW_background = dBm_to_W(np.max(data2, axis=1))*1e9
    Pth = np.mean(P_nW_background) + sigma_cutoff*np.std(P_nW_background)
    P_background = len(np.where(P_nW_background>Pth)[0]) / len(P_nW_background)

    all_t_load = [] if manual_var_vales is None else manual_var_vales
    all_P_signal = []
    for filename in tip_on_files: 
        file = load_h5_data(filename, base=data_dir)
        data = file['all_meas'] 
        P_nW = dBm_to_W(np.max(data, axis=1))*1e9
        P_signal = len(np.where(P_nW>Pth)[0]) / len(P_nW)
        if manual_var_vales is None:
            all_t_load.append(file[var]) 
        all_P_signal.append(P_signal) 
    idx = np.argsort(all_t_load)
    plot(np.array(all_t_load)[idx], np.array(all_P_signal)[idx], '.--', xlabel=xlabel, ylabel=f'Prob(max signal > {sigma_cutoff}$\sigma_n$)', label='Data')
    plt.axhline(P_background, linestyle='--', c='k', label='Background') 
    # plt.yscale('log')
    plt.legend() 
    plt.show()

def plot_peak_power_vs_var(tip_on_files, tip_off_files, data_dir, var='t_load', 
                     manual_var_vales=None,
                     sigma_cutoff=8,xlabel='Loading time (s)'):
    """
    Plots the probability of detected power exceeding a threshold for Tip-On and Tip-Off datasets across a specified variable.

    Parameters:
    - tip_on_files: List of file paths for Tip-On datasets.
    - tip_off_files: List of file paths for Tip-Off datasets.
    - data_dir: Base directory where the data files are located.
    - var: The variable to use for mapping datasets (default is 't_load').
    """
    # --- Helper functions (Ensure dBm_to_W, load_h5_data, and data_dir are defined) ---
    # def dBm_to_W(dbm): ...
    # def load_h5_data(filepath, base=None): ...
    # data_dir = '/home/electron/data/experiment_07142026'

    file2 = load_h5_data(tip_off_files[0], base=data_dir) 
    data2 = file2['all_meas'] 
    P_nW_background = dBm_to_W(np.max(data2, axis=1))*1e9
    Pth = np.mean(P_nW_background) + sigma_cutoff*np.std(P_nW_background)
    P_background_over = P_nW_background[np.where(P_nW_background>Pth)[0]]
    P_background = 0 if len (P_background_over) == 0 else np.mean(P_background_over)

    all_t_load = [] if manual_var_vales is None else manual_var_vales
    all_P_signal = []
    all_P_signal_std = []
    for filename in tip_on_files: 
        file = load_h5_data(filename, base=data_dir)
        data = file['all_meas'] 
        P_nW = dBm_to_W(np.max(data, axis=1))*1e9
        P_nW_sig = P_nW[np.where(P_nW>Pth)[0]]
        if len(P_nW_sig) > 0:
            P_signal = np.mean(P_nW_sig) 
            P_signal_std = np.std(P_nW_sig)
            P_signal_err = P_signal_std / np.sqrt(len(P_nW_sig))
        else: 
            P_signal_err = 0
            P_signal = 0
        if manual_var_vales is None:
            all_t_load.append(file[var]) 
        all_P_signal.append(P_signal) 
        all_P_signal_std.append(P_signal_err)
    idx = np.argsort(all_t_load)
    plot_errbar(np.array(all_t_load)[idx], np.array(all_P_signal)[idx], yerr=np.array(all_P_signal_std)[idx],fmt='.--', xlabel=xlabel, ylabel=f'Mean power above {sigma_cutoff}$\sigma_n$ threshold (nW)', label='Data')
    plt.axhline(P_background, linestyle='--', c='k', label='Background') 
    # plt.yscale('log')
    plt.legend() 
    plt.show()

def plot_power_by_rid(rid, search_dir, home_dir, threshold_nw=10.0, N_data_SWT=751, tmin=-1, tmax=1e6):
    # 1. Resolve file path using RID
    filename = find_file_by_rid(rid, search_dir, home_dir)
    if filename is None:
        print(f"Error: No HDF5 file found for RID {rid}")
        return

    # 2. Load dataset and metadata
    file = load_h5_data(filename, base=home_dir)['datasets']
    SWT = load_artiq_h5_arguments(filename, 'SSA_SWT', base=home_dir)
    
    all_powers = []

    # 3. Process and plot datasets passing threshold
    for key in file: 
        if key.startswith('all_meas'): 
            powers = file[key] 
            if len(powers) > 0:
                index = int(key[9:]) 
                time_stamp = file[f'time_stamps_{index}']
                plotted_any = False

                for j in range(len(time_stamp)): 
                    power_nW = 10**(powers[j]/10) * 1e6
                    
                    if np.max(power_nW) > threshold_nw:
                        t_ax = np.linspace(time_stamp[j], time_stamp[j] + SWT, N_data_SWT)
                        idx = np.where( (t_ax > tmin) & (t_ax < tmax))
                        plot(
                            t_ax[idx], 
                            power_nW[idx], 
                            xlabel='Time window (s)', 
                            ylabel='Detected Power (nW)', 
                            marker='.', 
                            title=f"RID {rid} - Dataset {key[9:]}"
                        )
                        plotted_any = True
                    
                    all_powers.extend(powers[j])
                
                if plotted_any:
                    plt.show()

def plot_power_by_rid_singleRow(rid, search_dir, home_dir, threshold_nw=10.0, N_data_SWT=751, tmin=-1, tmax=1e6):
    # 1. Resolve file path using RID
    filename = find_file_by_rid(rid, search_dir, home_dir)
    if filename is None:
        print(f"Error: No HDF5 file found for RID {rid}")
        return

    # 2. Load dataset and metadata
    file = load_h5_data(filename, base=home_dir)['datasets']
    SWT = load_artiq_h5_arguments(filename, 'SSA_SWT', base=home_dir)
    
    # 3. First pass: Collect all valid data curves that pass the threshold
    curves_to_plot = []

    for key in file: 
        if key.startswith('all_meas'): 
            powers = file[key] 
            if len(powers) > 0:
                index = int(key[9:]) 
                time_stamp = file[f'time_stamps_{index}']

                for j in range(len(time_stamp)): 
                    power_nW = 10**(powers[j]/10) * 1e6
                    
                    if np.max(power_nW) > threshold_nw:
                        t_ax = np.linspace(time_stamp[j], time_stamp[j] + SWT, N_data_SWT)
                        idx = np.where((t_ax > tmin) & (t_ax < tmax))
                        
                        curves_to_plot.append({
                            'time': t_ax[idx],
                            'power': power_nW[idx],
                            'title': f"RID {rid} - Dataset {index}"
                        })

    num_subplots = len(curves_to_plot)
    if num_subplots == 0:
        print(f"No datasets exceeded the {threshold_nw} nW threshold for RID {rid}.")
        return

    # 4. Create single figure with a dynamically sized 1-row layout
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4), sharey=True)
    
    # Ensure axes is iterable even if num_subplots == 1
    if num_subplots == 1:
        axes = [axes]

    # 5. Plot each collected curve into its respective subplot
    for ax, data in zip(axes, curves_to_plot):
        ax.plot(data['time'], data['power'], marker='.', linestyle='-')
        ax.set_title(data['title'])
        ax.set_xlabel('Time window (s)')
        ax.grid(True)

    axes[0].set_ylabel('Detected Power (nW)')
    plt.tight_layout()
    plt.show()

F_OUT_MHZ = float(os.environ.get("RFSOC_F_OUTPUT_MHZ", "552.96"))
def envelope_decimate(iq, t, n_points: int):
    """Min/max-per-bin downsample of a waveform for display: keeps narrow spikes
    that plain striding would drop. Returns ``(t_bins, lo, hi)``, each length
    <= n_points."""
    mag = np.hypot(iq[:, 0], iq[:, 1])
    # phase = np.angle(iq[:,0]+1j*iq[:,1])
    mag = np.asarray(mag); t = np.asarray(t)
    if len(mag) <= 2 * n_points:
        return t, mag, mag
    s = int(np.ceil(len(mag) / n_points))
    k = (len(mag) // s) * s
    m = mag[:k].reshape(-1, s)
    ireshape = iq[:,0][:k].reshape(-1,s) 
    qreshape = iq[:,1][:k].reshape(-1,s)
    
    return t[:k:s], m.min(1), m.max(1), m.mean(1), np.angle(ireshape.mean(1)+1j*qreshape.mean(1)), ireshape.mean(1), qreshape.mean(1) 

def plot_rfsoc(target_id, base_dir='/home/electron/data/RFSoc_data', ndecimated=1000, sharey=True):
    event_files = get_artiq_event_files(base_dir, target_id)
    event_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    nfig = len(event_files) 
    fig, ax = plt.subplots(ncols=nfig, figsize=(5*nfig,4), sharey=sharey)
    if nfig == 1: 
        ax = [ax]
    for i in range(nfig): 
        df = np.load(event_files[i])
        iq, amp, t = df['iq'], df['amp'], df['t_wall']
        fs_mhz = float(df["fs_msps"]) if "fs_msps" in getattr(df, "files", []) else F_OUT_MHZ
        t = np.arange(len(iq[:,0])) / fs_mhz                    # microseconds
        tb, lo, hi, mean, phase, imean, qmean = envelope_decimate(iq, t, int(2000)) 
        
        plot_ax(ax[i], tb/1e3, mean, '-', xlabel='t (ms)', title=f'Event {i+1}')
        # plot_ax(ax[1], tb/1e3, phase/np.pi, xlabel='t (ms)', ylabel=r'Phase ($\pi$)')
    ax[0].set_ylabel('Digital code (mean voltage)')
    plt.tight_layout() 
    plt.show()