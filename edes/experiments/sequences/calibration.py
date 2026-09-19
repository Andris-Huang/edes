from edes.experiments.sequences import base as sequences
from edes.utils.plotting import plot, plot_ax, big_plt_font, plot_ax_errbar, plot_errbar
import matplotlib.pyplot as plt
import numpy as np 

def reset_tip_V(exp, tip_current=2, V_pos=40, V_warning=1300, V_step=10, R=200e6, V_range=100, I_max = 30e-9, show_plot=False):
    exp.max_I = 1
    while exp.max_I < tip_current:
        tip_Vsweep = sequences.TipVoltageSweepDifferential(saving_dir=exp.saving_dir, 
                                         V_start=exp.V_tip-V_range, V_stop=exp.V_tip+V_step/2, V_step=V_step, R=R, 
                                         N_avg=1, t_PSU_settle=2, t_meas_delay=0.2, 
                                         V_fixed=V_pos, ch_sweep='neg', ch_fixed='pos',
                                         FEtip_PSU=exp.FEtip_PSU, multimeter=exp.Agilent, I_max=I_max)
        file = tip_Vsweep.run_save()
        I = np.mean(file['all_I'], axis=1)*1e9 
        V = file['all_V']
        if show_plot:
            plot_errbar(file['all_V'], np.mean(file['all_I'], axis=1)*1e9, yerr=np.std(file['all_I'], axis=1)*1e9, fmt='.--', 
                        xlabel='Tip voltage (-V)', ylabel='Electrode current (nA)')
            plt.show()
        exp.max_I = max(I) 
        if max(I) < tip_current-0.1: 
            exp.V_tip += 5
        else: 
            exp.V_tip = float(V[np.where(I-tip_current > -0.1)[0][0]])
        if exp.V_tip >= V_warning: 
            print(f'>>> Tip voltage exceeding safe level at {V_warning}V, setting to max voltage')
            exp.V_tip = V_warning