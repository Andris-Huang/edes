import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import numpy as np
import os
import datetime

from edes.modules.VNA.vna_utils import take_vna_sweep, query_temp

# Global mock temp variable for testing
MOCK_TEMP = 3.0 

# --- MAIN GUI APPLICATION ---
class CryoMeasurementApp:
    def __init__(self, root, temp_list=[]):
        self.root = root
        self.root.title("Cryostat VNA Automation")
        self.root.geometry("1000x700")

        # --- Control Variables ---
        self.is_running = False
        self.temp_list = temp_list
        self.current_target_index = 0
        self.data_folder = os.getcwd() # Default to current dir
        
        # --- GUI Layout ---
        self.setup_ui()
        
        # --- Plotting Setup ---
        self.setup_plot()

    def setup_ui(self):
        # Top Frame for Inputs
        control_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Temp List Input
        ttk.Label(control_frame, text="Target Temps (comma sep):").grid(row=0, column=0, sticky="w")
        self.entry_temps = ttk.Entry(control_frame, width=40)
        self.entry_temps.insert(0, "4, 10, 20, 50, 100") # Default values
        self.entry_temps.grid(row=0, column=1, padx=5, pady=5)

        # Folder Input
        ttk.Label(control_frame, text="Data Folder:").grid(row=1, column=0, sticky="w")
        self.entry_folder = ttk.Entry(control_frame, width=40)
        self.entry_folder.insert(0, self.data_folder)
        self.entry_folder.grid(row=1, column=1, padx=5, pady=5)

        # Buttons
        self.btn_start = ttk.Button(control_frame, text="Start Measurement", command=self.start_process)
        self.btn_start.grid(row=0, column=2, rowspan=2, padx=20, sticky="nsew")
        
        self.btn_stop = ttk.Button(control_frame, text="STOP", command=self.stop_process, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=3, rowspan=2, padx=5, sticky="nsew")

        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.lbl_current_temp = ttk.Label(status_frame, text="Current Temp: -- K", font=("Arial", 12, "bold"))
        self.lbl_current_temp.pack(side=tk.LEFT, padx=20)
        
        self.lbl_status = ttk.Label(status_frame, text="Status: Idle", foreground="blue")
        self.lbl_status.pack(side=tk.LEFT, padx=20)

    def setup_plot(self):
        # Matplotlib Figure embedded in Tkinter
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.ax.set_title("VNA Measurement Live View")
        self.ax.set_xlabel("Frequency (GHz)")
        self.ax.set_ylabel("|S| (dB)")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_process(self):
        try:
            # Parse temperature list
            raw_temps = self.entry_temps.get()
            self.temp_list = [float(t.strip()) for t in raw_temps.split(',')]
            self.data_folder = self.entry_folder.get()
            
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)
                
            self.is_running = True
            self.current_target_index = 0
            
            # UI Updates
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.lbl_status.config(text="Starting process...", foreground="green")
            
            # Start Background Thread
            self.thread = threading.Thread(target=self.measurement_loop)
            self.thread.daemon = True # Kills thread if app closes
            self.thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid temperature list format.")

    def stop_process(self):
        self.is_running = False
        self.lbl_status.config(text="Stopping...", foreground="red")

    def update_gui_temp(self, temp_val):
        """Thread-safe update of the temperature label"""
        self.lbl_current_temp.config(text=f"Current Temp: {temp_val:.3f} K")

    def update_gui_status(self, msg):
        """Thread-safe update of the status label"""
        self.lbl_status.config(text=msg)

    def update_plot(self, freq, S11, S21, temp_val):
        """Thread-safe plot update"""
        self.ax.clear()
        
        # Plot S11
        self.ax.plot(freq/1e9, 20*np.log10(np.abs(S11)), label=f'S11 ({temp_val}K)', color='b')
        # Plot S21
        self.ax.plot(freq/1e9, 20*np.log10(np.abs(S21)), label=f'S21 ({temp_val}K)', color='r')
        
        self.ax.set_title(f"Last Measurement: {temp_val} K")
        self.ax.set_xlabel('Frequency (GHz)')
        self.ax.set_ylabel('|S| (dB)')
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()

    def measurement_loop(self):
        """The background logic that waits and measures"""
        
        for target_temp in self.temp_list:
            if not self.is_running:
                break
                
            self.root.after(0, self.update_gui_status, f"Waiting for Temp > {target_temp} K")
            
            # --- WAITING LOOP ---
            while self.is_running:
                # 1. Query Temp
                try:
                    # Note: '4K' in your example is usually the sensor name
                    current_temp = query_temp('4K') 
                except Exception as e:
                    print(f"Error reading temp: {e}")
                    current_temp = 0
                
                # Update GUI with current temp
                self.root.after(0, self.update_gui_temp, current_temp)
                
                # 2. Check Condition (Surpasses / Greater than)
                # Assuming a warming up experiment where we wait for temp to rise above target
                if current_temp >= target_temp:
                    self.root.after(0, self.update_gui_status, f"Triggered! Measuring at {current_temp:.2f} K")
                    
                    # 3. Take Measurement
                    timestamp = datetime.datetime.now().strftime("%H%M%S")
                    filename = f"Temp_{current_temp:.2f}K_{timestamp}"
                    
                    S11, S21, freq = take_vna_sweep(
                        filename, 
                        self.data_folder, 
                        1.4e9, 
                        1.7e9, 
                        N_points=2000
                    )
                    
                    # 4. Update Plot on Main Thread
                    self.root.after(0, self.update_plot, freq, S11, S21, current_temp)
                    
                    # Measurement done, break wait loop to move to next target
                    break 
                
                # Wait 10s before checking again
                time.sleep(10) # Reduced to 1s for testing, change back to 10s
                
        # End of list
        self.is_running = False
        self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
        self.root.after(0, self.update_gui_status, "Measurement Sequence Complete.")

# --- RUN APP ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CryoMeasurementApp(root, temp_list=[1.0, 1.5, 2.0, 2.5])
    root.mainloop()