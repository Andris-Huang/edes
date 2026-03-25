import csv
import tkinter as tk
from tkinter import ttk
import time
from collections import deque
import matplotlib
import numpy as np
from datetime import datetime

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from lakeshore import Model240, Model240InputParameter


# =========================
# Lake Shore Initialization
# =========================
my_model_240 = Model240()

rtd_config = Model240InputParameter(
    my_model_240.SensorTypes.DIODE,
    False,
    False,
    my_model_240.Units.SENSOR,
    True,
    my_model_240.InputRange.RANGE_DIODE
)

for channel in range(1, 3):
    my_model_240.set_input_parameter(channel, rtd_config)


# =========================
# GUI Application
# =========================
class TemperatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lake Shore Model 240 Temperature Monitor")

        self.is_dark_mode = False  # Start in light mode

        # -------------------------
        # Buffers
        # -------------------------
        self.start_time = time.time()

        # 10 minutes @ 10 points/min
        self.short_points = 100
        self.short_time = np.linspace(0, 10, self.short_points)
        self.short_ch1 = deque(maxlen=self.short_points)
        self.short_ch2 = deque(maxlen=self.short_points)

        # 12 hours @ 10 points/hour
        self.long_points = 120
        self.long_time = np.linspace(0, 12, self.long_points)
        self.long_ch1 = deque(maxlen=self.long_points)
        self.long_ch2 = deque(maxlen=self.long_points)

        self.sec_counter = 0
        self.min_counter = 0

        # -------------------------
        # Labels & Dark Mode Toggle
        # -------------------------
        label_frame = ttk.Frame(root)
        label_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Temperature labels (separate columns, centered together)
        self.ch1_label = ttk.Label(label_frame, text="CH1: -- K", font=("Arial", 14))
        self.ch1_label.grid(row=0, column=4, padx=20, sticky='nsew')
        
        self.ch2_label = ttk.Label(label_frame, text="CH2: -- K", font=("Arial", 14))
        self.ch2_label.grid(row=0, column=5, padx=20, sticky='nsew')
        
        # Dark mode button (right-aligned in its column)
        self.toggle_button = ttk.Button(label_frame, text="Toggle Dark Mode", command=self.toggle_dark_mode)
        self.toggle_button.grid(row=0, column=9, padx=20, sticky='e')
        
        # Make columns stretch equally, ensuring labels are centered
        label_frame.columnconfigure(0, weight=1)
        label_frame.columnconfigure(1, weight=1)
        label_frame.columnconfigure(9, weight=1)
        # label_frame.columnconfigure(4, weight=1)
        # label_frame.columnconfigure(7, weight=0)  # Keep the toggle button column non-stretchable



        # -------------------------
        # Plot Frame
        # -------------------------
        plot_frame = ttk.Frame(root)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # ===== Short-term Figure =====
        self.fig_short = Figure(figsize=(6, 4), dpi=100)
        self.ax_short = self.fig_short.add_subplot(111)
        self.ax_short.set_title("Last 10 Minutes")
        self.ax_short.set_xlabel("Time (min)")
        self.ax_short.set_ylabel("Temperature (K)")
        self.ax_short.set_xlim(0, 10)
        self.ax_short.grid(True)

        self.short_line1, = self.ax_short.plot([], [], label="CH1")
        self.short_line2, = self.ax_short.plot([], [], label="CH2")
        self.ax_short.legend()

        self.canvas_short = FigureCanvasTkAgg(self.fig_short, master=plot_frame)
        self.canvas_short.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ===== Long-term Figure =====
        self.fig_long = Figure(figsize=(6, 4), dpi=100)
        self.ax_long = self.fig_long.add_subplot(111)
        self.ax_long.set_title("Last 12 Hours")
        self.ax_long.set_xlabel("Time (hours)")
        self.ax_long.set_ylabel("Temperature (K)")
        self.ax_long.set_xlim(0, 12)
        self.ax_long.grid(True)

        self.long_line1, = self.ax_long.plot([], [], label="CH1")
        self.long_line2, = self.ax_long.plot([], [], label="CH2")
        self.ax_long.legend()

        self.canvas_long = FigureCanvasTkAgg(self.fig_long, master=plot_frame)
        self.canvas_long.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        plot_frame.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(1, weight=1)

        # Initialize style for ttk widgets
        self.style = ttk.Style()
        self.style.configure("DarkButton.TButton", background="black", foreground="white")
        self.style.configure("LightButton.TButton", background="white", foreground="black")

        # Start update loop
        self.update_temperature()

    # -------------------------
    # Update Loop
    # -------------------------
    def update_temperature(self):
        try:
            ch1K = my_model_240.get_kelvin_reading(1)
            ch2K = my_model_240.get_kelvin_reading(2)

            self.ch1_label.config(text=f"CH1: {ch1K:.3f} K")
            self.ch2_label.config(text=f"CH2: {ch2K:.3f} K")

            self.sec_counter += 1
            self.min_counter += 1

            # ----- Short-term: every 6 seconds -----
            if self.sec_counter >= 6:
                self.sec_counter = 0

                # Relative time: newest data at the right
                self.short_ch1.append(ch1K)
                self.short_ch2.append(ch2K)
                self.update_short_plot()

            

            # ----- Long-term: every 6 minutes -----
            if self.min_counter >= 360:
                self.min_counter = 0

                self.long_ch1.append(ch1K)
                self.long_ch2.append(ch2K)
                self.update_long_plot()


                #with open("/home/electron-cryo/Documents/cryo-data/01_22_2026_cooldown_temperature/log.csv", "a", newline="") as file:
                    #writer = csv.writer(file)
                    #writer.writerow([datetime.now().isoformat(), ch1K,ch2K])
                




        except Exception as e:
            print("Temperature read error:", e)

        self.root.after(1000, self.update_temperature)

    # -------------------------
    # Plot Updates
    # -------------------------
    def update_short_plot(self):
        self.short_line1.set_data(self.short_time[:len(self.short_ch1)], self.short_ch1)
        self.short_line2.set_data(self.short_time[:len(self.short_ch2)], self.short_ch2)

        temps = list(self.short_ch1) + list(self.short_ch2)
        if temps:
            self.ax_short.set_ylim(min(temps) - 0.5, max(temps) + 0.5)

        self.canvas_short.draw_idle()

    def update_long_plot(self):
        self.long_line1.set_data(self.long_time[:len(self.long_ch1)], self.long_ch1)
        self.long_line2.set_data(self.long_time[:len(self.long_ch2)], self.long_ch2)

        temps = list(self.long_ch1) + list(self.long_ch2)
        if temps:
            self.ax_long.set_ylim(min(temps) - 0.5, max(temps) + 0.5)

        self.canvas_long.draw_idle()

    # -------------------------
    # Dark Mode Toggle
    # -------------------------
    def toggle_dark_mode(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_mode:
            # Dark Mode
            self.root.configure(bg='black')

            # Dark Mode for the figures
            self.fig_short.patch.set_facecolor('black')
            self.fig_long.patch.set_facecolor('black')
            self.ax_short.set_facecolor('black')
            self.ax_long.set_facecolor('black')
            self.ax_short.tick_params(axis='x', colors='white')
            self.ax_short.tick_params(axis='y', colors='white')
            self.ax_long.tick_params(axis='x', colors='white')
            self.ax_long.tick_params(axis='y', colors='white')

            # Set title and label colors for Dark Mode
            self.ax_short.set_title("Last 10 Minutes", color='white')
            self.ax_short.set_xlabel("Time (min)", color='white')
            self.ax_short.set_ylabel("Temperature (K)", color='white')

            self.ax_long.set_title("Last 12 Hours", color='white')
            self.ax_long.set_xlabel("Time (hours)", color='white')
            self.ax_long.set_ylabel("Temperature (K)", color='white')

            # Set legend font color to white in dark mode
            # self.ax_short.legend(fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
            # self.ax_long.legend(fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')

            self.toggle_button.config(style='DarkButton.TButton')
            self.short_line1.set_color('cyan')   # Light C1
            self.short_line2.set_color('yellow')   # Light C1
            self.long_line1.set_color('cyan')   # Light C1
            self.long_line2.set_color('yellow')   # Light C1
            self.ax_long.legend(labelcolor='white',facecolor='black', edgecolor='white')
            self.ax_short.legend(labelcolor='white',facecolor='black', edgecolor='white')

        else:
            # Light Mode
            self.root.configure(bg='white')

            # Light Mode for the figures
            self.fig_short.patch.set_facecolor('white')
            self.fig_long.patch.set_facecolor('white')
            self.ax_short.set_facecolor('white')
            self.ax_long.set_facecolor('white')
            self.ax_short.tick_params(axis='x', colors='black')
            self.ax_short.tick_params(axis='y', colors='black')
            self.ax_long.tick_params(axis='x', colors='black')
            self.ax_long.tick_params(axis='y', colors='black')

            # Set title and label colors for Light Mode
            self.ax_short.set_title("Last 10 Minutes", color='black')
            self.ax_short.set_xlabel("Time (min)", color='black')
            self.ax_short.set_ylabel("Temperature (K)", color='black')

            self.ax_long.set_title("Last 12 Hours", color='black')
            self.ax_long.set_xlabel("Time (hours)", color='black')
            self.ax_long.set_ylabel("Temperature (K)", color='black')

            # Set legend font color to black in light mode
            self.ax_short.legend(fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')
            self.ax_long.legend(fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')

            self.toggle_button.config(style='LightButton.TButton')
            self.short_line1.set_color('C0')   # Light C1
            self.short_line2.set_color('C1')   # Light C1
            self.long_line1.set_color('C0')   # Light C1
            self.long_line2.set_color('C1')   # Light C1
            self.ax_long.legend()
            self.ax_short.legend()

        self.canvas_short.draw_idle()
        self.canvas_long.draw_idle()


# =========================
# Run
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = TemperatureGUI(root)
    root.mainloop()
