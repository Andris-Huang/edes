import numpy as np

def get_vctrl(p_out_desired_dbm, P_in=5.0):
    """
    Translates a desired output power to the required control voltage.
    Assumes a constant input power of 5 dBm.
    """
    # P_IN = 5.0 # dBm
    
    # Required S21 to achieve the desired output power
    s21_required = p_out_desired_dbm - P_in
    
    # Data extracted from the S21 vs Vctrl plot
    # Arrays must be monotonically increasing for numpy.interp to work properly on the x-axis
    s21_data = np.array([-40.5, -31.0, -21.0, -17.0, -15.0, -13.2, -12.0, -11.0, 
                         -10.0, -9.5, -8.8, -8.2, -7.6, -7.0, -6.4, -5.8, -5.2, -4.8])
    
    vctrl_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
                           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
    
    # Check if the requested power is physically possible with this attenuator
    if s21_required < s21_data[0]:
        raise ValueError(f"Desired P_out is too low. Minimum achievable S21 is {s21_data[0]} dB.")
    if s21_required > s21_data[-1]:
        raise ValueError(f"Desired P_out is too high. Maximum achievable S21 is {s21_data[-1]} dB.")
        
    # Interpolate to find the exact Vctrl
    vctrl_target = np.interp(s21_required, s21_data, vctrl_data)
    
    return vctrl_target