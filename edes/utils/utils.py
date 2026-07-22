import numpy as np
try:
    import sounddevice as sd
except: 
    pass
import jupyter_beeper
import time
# import tqdm 
# import sounddevice as sd
import subprocess
from scipy.constants import m_e, e

def beep(frequency=440, duration_seconds=0.5, volume=1):
    """
    Generates and plays a pure sine wave tone without relying on audio files.
    
    :param frequency: Pitch in Hz (e.g., 440 is Middle A)
    :param duration_seconds: How long the beep lasts
    :param volume: Float between 0.0 (silent) and 1.0 (max)
    """
    sample_rate = 44100  # Standard audio sample rate
    
    # Generate the time axis array
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    
    # Generate a pure sine wave formula: y = sin(2 * pi * f * t)
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Adjust volume and ensure data type is float32 for sounddevice
    audio_signal = (tone * volume).astype(np.float32)
    
    # Play the array directly over your Linux sound system
    sd.play(audio_signal, sample_rate)
    sd.wait() # Blocks execution until the beep finishes playing

def beep_jupyter(sequence='EE0E0CE0G000g'): 
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

def beep_VS(sequence='EE0E0CE0G000g'): 
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
            beep(frequency=freq_table[i], duration_seconds=0.1)
        time.sleep(0.05)


def beep_python():
    command = [
        "/home/electron/miniconda3/envs/e/bin/python", 
        "-c", 
        "from edes.utils.utils import beep_VS; beep_VS()"
    ]
    subprocess.run(command)
           

def homogenize_axis(arr, fill_value=np.nan):
    """
    Converts a ragged NumPy array (dtype=object) into a homogeneous,
    standard numeric ndarray by padding the last axis (-1) to the maximum length.
    """
    # Store the original shape of the leading dimensions
    orig_shape = arr.shape
    
    # Flatten the leading dimensions so we can easily iterate over the final axis elements
    flat_arr = arr.ravel()
    
    # Find the maximum length along the final axis
    max_len = max(len(np.atleast_1d(x)) for x in flat_arr)
    
    # Initialize a clean, homogeneous flat array with the fill value
    # Choose a float dtype if you are using np.nan
    homogeneous_flat = np.full((len(flat_arr), max_len), fill_value=fill_value, dtype=np.float64)
    
    # Fill the new array with the original data
    for i, sub_arr in enumerate(flat_arr):
        sub_arr = np.atleast_1d(sub_arr)
        homogeneous_flat[i, :len(sub_arr)] = sub_arr
        
    # Reshape back to the original leading dimensions, adding the new uniform last axis
    new_shape = orig_shape + (max_len,)
    return homogeneous_flat.reshape(new_shape)

# beep_python()

def U2_to_MHz(U2):
    return np.sqrt(abs(U2)*e*np.sqrt(20/16/np.pi)/(m_e/2)*1e6)/2/np.pi/1e6