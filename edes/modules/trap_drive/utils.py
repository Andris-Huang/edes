import jupyter_beeper
import time
import numpy as np

def beep(sequence='EE0E0CE0G00g'): 
    b = jupyter_beeper.Beeper()
    freq_table = {'C': 261.63,
                  'D': 293.66,
                  'E': 329.63,
                  'F': 349.23,
                  'G': 392.00,
                  'A': 440.00,
                  'B': 493.88, 
                  'g': 196}
    for i in sequence:
        if i == '0': 
            time.sleep(0.1)
        else: 
            b.beep(frequency=freq_table[i], secs=0.1)
        time.sleep(0.12)


def V_to_P(V): 
    P = 10*np.log10(V**2/100*1000) 
    return P

def P_to_V(P): 
    V = np.sqrt(10**(P/10)/1000*100) 
    return V