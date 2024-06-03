import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
from tqdm import trange
import scipy as sp

########## Constants ##########
qe = -spc.e
e = spc.e
m = spc.m_e
kB = spc.k


########## Plotting functions for laziness ##########
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



########## Circuit basics functions ##########

def parallel(a, b):
    """
    The parallel impedance of two individual impedances
    """
    return 1/(1/a+1/b)

def W_to_dBm(W):
    return 10*np.log10(1000*W)

def dBm_to_W(dBm):
    return 10**(dBm/10)/1000

def dB_to_frac(dB):
    return 10**(dB/10)

def frac_to_dB(frac):
    return 10*np.log10(frac)

def voltage_noise_to_dBm(V, BW_meas=100, Z_meas=50):
    """
    V/sqrt(Hz) to dBm converter

    Parameters
    ----------
    V : [V/sqrt(Hz)]
        Voltage noise
    BW_meas : [Hz]
        measurement bandwidth, in Hz
    Z_measure : [Ohm]
        measurement impedance, in Ohm

    Returns
    -------
    Noise power in dBm
    """
    return W_to_dBm((V)**2*BW_meas/Z_meas)


########## Additional useful functions ##########
def find_width(x, y, threshold=0.5):
    """
    Find the width of a signal, width is defined as the
    distance between two locations where the corresponding
    y values are equal to threshold. If y has range between 
    0 and 1, and threshold is set to 0.5, then this finds
    the full-width-half-maximum.
    """
    i_min = np.argmin(y)
    i_max1 = np.argmax(y[:i_min])
    i_max2 = np.argmax(y[i_min:])+i_min
    i_thres1 = np.argmin(abs(y[i_max1:i_min]-threshold)) + i_max1
    i_thres2 = np.argmin(abs(y[i_min:i_max2]-threshold)) + i_min
    return x[i_thres2] - x[i_thres1]

def moving_avg(y, window, display_progress=True):
    """
    Calculate the moving average of y values based on the specified window size.
    
    Parameters
    ----------
        x (list or numpy array): The x values.
        y (list or numpy array): The y values.
        window (int): The size of the moving average window.
        
    Returns
    -------
        The moving average of y values.
    """
    moving_avg = []
    for i in trange(len(y), desc='Measuring Signal', disable=(not display_progress)):
        if i >= window - 1:
            avg = sum(y[i - window + 1:i + 1]) / window
            moving_avg.append(avg)
        else:
            moving_avg.append(sum(y[:i + 1])/(i+1))
    return np.array(moving_avg)


########## Functions for LC circuits, particularly for electron and the tank circuit ##########
def le(d, n, m=m, e=e):
    """
    equivalent inductance of electron

    Parameters
    ----------
    d : [m]
        Effective distance
    n : int
        Number of electrons
    """
    return m*d**2/e**2/n

def ce(omega, l, m=m, e=e):
    """
    equivalent capacitance of electron, calculated based on electron inductance

    Parameters
    ----------
    omega : [rad/s]
        Motional frequency of the electron, in angular frequency
    l : [H]
        Equivalent inductance of the electron
    """
    return 1/(omega**2*l)

def get_lc(d, n, omega, m=m, e=e):
    """
    Get the equivalent lc values of the electron

    Parameters
    ----------
    d : [m]
        Effective distance
    n : int
        Number of electrons
    omega : [rad/s]
        Motional frequency of the electron, in angular frequency

    Returns
    -------
    l : [H]
        Equivalent inductance of the electron
    c : [F]
        Equivalent capacitance of the electron
    """
    l = le(d, n, m, e)
    c = ce(omega, l, m, e)
    return l, c

def Rt(Q, Z0):
    """
    The equivalent lumped real impedance of 
    the tank circuit
    """
    return Q*Z0

def Ct(omega, Q, R):
    """
    The equivalent capacitance of the tank circuit
    """
    return Q/(omega*R)

def Lt(omega, Q, R):
    """
    The equivalent inductance of the tank circuit
    """
    return R/(omega*Q)

def get_RLC(Q, Z0, omega):
    """
    Get the equivalent RLC values of the tank circuit
    given the Q, Z0, and resonant frequency.
    
    Parameters
    ----------
    Q : float
        Quality factor
    Z0 : [Ohm]
        Characteristic impedance of the tank circuit,
        or equivalently Z0 = sqrt(L/C)
    omega : [rad/s]
        Resonant frequency of the tank circuit, 
        in angular frequency

    Returns
    -------
    R : [Ohm]
        Equivalent real impedance of the tank circuit
    L : [H]
        Equivalent inductance
    C : [F]
        Equivalent capcitance
    """
    R = Rt(Q, Z0)
    L = Lt(omega, Q, R)
    C = Ct(omega, Q, R)
    return R, L, C


########## Circuit elements ##########
class Component:
    """
    Base class for a circuit component used. Has some 
    common functions needed for all circuit elements.
    """
    def __init__(self, name='Device', **kwargs):
        self.name = name
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    def H(self, omega):
        """
        Transfer function or frequency resoponse.
        V_out(omega) = H(omega)V_in(omega)
        
        Parameters
        ----------
        omega : ndarray
            Angular frequency.
        
        Returns
        -------
        H : ndarrar
            Linear circuit frequency response.
        """
        raise NotImplementedError
    
    def H_mag(self, omega):
        """
        Returns the magnitude of the frequency response.
        """
        return abs(self.H(omega))
    
    def get_noise(self, f):
        """
        Returns the power spectral density of the noise
        generated by the device.
        """
        return 0

    def white_noise(self, f, PSD):
        """
        Generate random white noise in freq domain 
        based on the length of f and the variance PSD.
        """
        return PSD * np.random.normal(1, 1, np.shape(f))
    
    def __call__(self, f, signal):
        """
        Returns the magnitude square of the output siganl given 
        input signal and frequencies.
        
        Parameters
        ----------
        f : ndarray
            Frequency in Hz
        signal : ndarray
            Values of the signal
        
        Returns
        -------
        V2_out : ndarray
            The magnitude of the output signal. Default assumes 
            linear frequency response, hence 
            |V_out|^2 = |H(omega)|^2 |V_in|^2
        noise : ndarray
            The noise psd of a given device.
        """
        omega = 2*np.pi*f
        return abs(signal)*self.H_mag(omega)**2


class BandPassFilter(Component):
    def __init__(self, omega_0, Q, name='BPF'):
        """
        A band-pass filter.
        
        Parameters
        ----------
        omega_0 : [rad/s]
            Resonant frequency of the filter, in angular frequency
        Q : float
            Quality factor
        """
        super().__init__(name=name, omega_0=omega_0, Q=Q)
    
    def H(self, omega):
        s = 1j*omega
        Q = self.Q
        omega_0 = self.omega_0
        return omega_0/Q*s/(s**2+omega_0/Q*s+omega_0**2)


class SpectrumAnalyzer(Component):
    def __init__(self, noise_psd, BW_measurement, Z_measurement, 
                 display_progress=True, name='Spectrum Analyzer'):
        """
        A spectrum analyzer that measures the power spectrum in dBm
        and averages the signal through a moving average filter of 
        a window size defined by measurement bandwidth.

        Parameters
        ----------
        BW_measurement : [Hz]
            Measurement bandwidth of the spectrum analyzer
        Z_measurement : [Ohm]
            The measurement impedance of the spectrum analyzer
        noise_floor : [V^2/Hz]
            The noise power spectral density of the spectrum analyzer
        display_progress : bool
            Whether to display a progress bar when doing 
            moving average on the signal
        """
        super().__init__(name=name, noise_psd=noise_psd, 
                         Z_measurement=Z_measurement, 
                         BW_measurement=BW_measurement,
                         display_progress=display_progress)

    def __call__(self, f, signal):
        window_size = int(self.BW_measurement/(f[1]-f[0]))
        noise_floor = self.white_noise(f, self.noise_psd)
        output_signal = (signal + noise_floor)*self.BW_measurement/self.Z_measurement  # Power spectrum
        self.noise_floor = W_to_dBm(noise_floor*self.BW_measurement)
        if window_size <= 1:
            return W_to_dBm(output_signal)
        else:
            return W_to_dBm(moving_avg(output_signal, window_size, display_progress=self.display_progress))

            
class HarmonicElectron(Component):
    def __init__(self, n, d, omega, m=m, e=e, name='Electron'):
        """
        A harmonic electron modeled as a series-lc circuit. Can
        use electron.l and electron.c to see the value of equivalent
        l and c after initializing
            electron = HarmonicElectron(n, d, omega, m, e)
        
        Parameters
        ----------
        n : int
            Number of electrons 
        d : [m]
            Effective distance
        omega : [rad/s]
            Motional frequency of the electron, in angular frequency
        m : [kg]
            Mass of electron
        e : [c]
            Magnitude of the charge of electron
        """
        super().__init__(name=name, n=n, d=d, omega=omega, m=m, e=e)
        self.l, self.c = get_lc(d, n, omega, m, e)

    def get_lc(self):
        return self.l, self.c

        
class TankCircuit(Component):
    def __init__(self, Q, Z0, omega_0, T=4, name='TankCircuit'):
        """
        Tank circuit.

        Parameters
        ----------
        Q : int
            Quality factor of tank circuit
        Z0 : [Ohm]
            Characteristic impedance of the tank circuit
        omega_0 : [rad/s]
            Resonant frequency of the tank circuit, 
            in angular frequency
        """
        super().__init__(name=name, Q=Q, Z0=Z0, omega_0=omega_0, T=T)
        self.R, self.L, self.C = get_RLC(Q, Z0, omega_0)
        self.T = T

    def get_RLC(self):
        return self.R, self.L, self.C

    def get_noise(self, f):
        """
        Returns the power spectral density of the noise.
        """
        PSD = 4*kB*self.T*self.R
        return self.white_noise(f, PSD)

        
class HarmoincElectronCoupledTank(Component):
    def __init__(self, n, d, Q, Z0, omega_0, omega_z, 
                 m=m, e=e, T=4, name='Electron'):
        super().__init__(name=name, n=n, d=d, Q=Q, Z0=Z0,
                         omega_0=omega_0, omega_z=omega_z,
                         T=T, m=m, e=e)
        """
        Initializing a series lc (electron) coupled to a parallel RLC (tank circuit)
        circuit, see Wineland 1975.
        
        Parameters
        ----------
        n : int
            Number of electrons 
        d : [m]
            Effective distance
        m : [kg]
            Mass of electron
        e : [c]
            Magnitude of the charge of electron
        Q : int
            Quality factor of tank circuit
        Z0 : [Ohm]
            Characteristic impedance of the tank circuit
        omega_0 : [rad/s]
            Resonant frequency of the tank circuit, 
            in angular frequency
        omega_z : [rad/s]
            Motional frequency of the electron, in angular frequency
        T : [K]
            Temperature of the tank circuit
        """
        self.Electron = HarmonicElectron(n, d, omega_z, m, e, name)
        self.TankCircuit = TankCircuit(Q, Z0, omega_0, T, name)
        self.R, self.L, self.C = self.TankCircuit.get_RLC()
        self.l, self.c = self.Electron.get_lc()

    def H(self, omega):
        '''
        series lc (electron) coupled to a parallel RLC (tank circuit) circuit, see Wineland 1975
        '''
        s = 1j*omega
        s = 1j*omega
        Zl = s*self.l
        ZL = s*self.L
        Zc = 1/(s*self.c)
        ZC = 1/(s*self.C)
        R = self.R
        
        Zlc = parallel( (Zl+Zc), parallel(ZL, ZC) )
        return Zlc / (Zlc+R)

        
class ElectronCoupledToTank(Component):
    def __init__(self, Electron, TankCircuit, name='Tank Circuit'):
        """
        Initializing a circuit where an Electron is coupled to a TankCircuit.
        This differ from the HarmoincElectronCoupledTank component as the 
        current component directly passes in Electron and TankCircuit as arguments,
        so they need to be already initialized elsewhere.
        """
        super().__init__(name=name, Electron=Electron, TankCircuit=TankCircuit)
        self.Electron = Electron
        self.TankCircuit = TankCircuit
        self.l, self.c = Electron.get_lc()
        self.R, self.L, self.C = TankCircuit.get_RLC()

    def H(self, omega):
        s = 1j*omega
        s = 1j*omega
        Zl = s*self.l
        ZL = s*self.L
        Zc = 1/(s*self.c)
        ZC = 1/(s*self.C)
        R = self.R
        
        Zlc = parallel( (Zl+Zc), parallel(ZL, ZC) )
        return Zlc / (Zlc+R)


class Amplifier(Component):
    def __init__(self, gain, voltage_noise, name='Amplifier'):
        """
        A generic amplifier.
        
        Parameters
        ----------
        gain : [Decimal]
            Gain factor of the amplifier, NOT in dB
        voltage_noise : [V/sqrt(Hz)]
            The voltage noise spectral density. The power spectral
            density (PSD) can be obtained by voltage_noise^2 (V^2/Hz)
        """
        super().__init__(name=name, gain=gain, voltage_noise=voltage_noise)

    def get_noise(self, f):
        psd = self.voltage_noise**2
        return self.white_noise(f, psd)

    def __call__(self, f, signal):
        """
        We here assumes the noise is input-referred, so the signal and noise
        will both be amplified by the gain.
        """
        noise = self.get_noise(f)
        return self.gain**2 * (signal + noise)

        
class DetectionSetup(Component):
    def __init__(self, BW_measurement, Z_measurement, noise_floor, 
                 display_progress=False, name='DetectionSetup', 
                 **kwargs):
        """
        A detection setup base class. We here setup two spectrum analyzers
        as a comparison, one with a given noise floor and the other without.
        This can inform us whether the signal is above the realistic noise floor
        of the measurement device.
        """
        super().__init__(name=name, BW_measurement=BW_measurement, 
                         Z_measurement=Z_measurement, **kwargs)
        self.spectrum_analyzer = SpectrumAnalyzer(noise_floor, BW_measurement, Z_measurement, 
                                                  display_progress=display_progress)
        self.noiseless_spectrum_analyzer = SpectrumAnalyzer(0, BW_measurement, Z_measurement, 
                                                            display_progress=False)
        
    def measure_spectrum(self, freq, psd_signal):
        """
        Measure the power spectrum of the signal using a spectrum analyzer
        with a defined noise floor and an ideal spectum analyzer with 0 noise.
        """
        ideal_spectrum = self.noiseless_spectrum_analyzer(freq, psd_signal)
        real_spectrum = self.spectrum_analyzer(freq, psd_signal)
        return ideal_spectrum, real_spectrum
        

class AnharmonicElectronSetup(DetectionSetup):
    def __init__(self, n, d, Q, Z0, f0, fz_0, delta_fz, T, 
                 BW_measurement, Z_measurement, noise_floor, 
                 N_average, m=m, e=e,
                 devices=[],
                 display_progress=False, 
                 name='AnharmonicElectron'):
        """
        The electron coupled to a tank circuit, with a sequence of devices
        applied in series to the signal after tank circuit. A finite width
        is assumed and the electron's motional spectrum has a Gaussian 
        distribution with width delta_fz (std = delta_fz/2).

        Parameters
        ----------
        n : int
            Number of electrons 
        d : [m]
            Effective distance
        m : [kg]
            Mass of electron
        e : [c]
            Magnitude of the charge of electron
        Q : int
            Quality factor of tank circuit
        Z0 : [Ohm]
            Characteristic impedance of the tank circuit
        f0 : [Hz]
            Resonant frequency of the tank circuit, 
            in Hz (NOT angular freq)
        fz_0 : [Hz]
            Motional frequency of the electron (resonant 
            freq of the series lc), in Hz (NOT angular freq)
        delta_fz : [Hz]
            Width of the motional frequency of the electron,
            where the motional frequency is assumed to have a
            Gaussian distribution with std = width/2
        T : [K]
            Temperature of the tank circuit
        BW_measurement : [Hz]
            Measurement bandwidth of the spectrum analyzer
        Z_measurement : [Ohm]
            The measurement impedance of the spectrum analyzer
        noise_floor : [V^2/Hz]
            The noise power spectral density of the spectrum analyzer
        N_average : int
            The number of shots to take and do averaging on
        devices : list
            A list of devices in the detection setup, e.g. 
            amplifiers, filters, etc.
        display_progress : bool
            Whether to display a progress bar when doing 
            moving average on the signal
        """
        super().__init__(BW_measurement, Z_measurement, noise_floor, 
                         name=name, n=n, d=d, Q=Q, Z0=Z0, f0=f0, 
                         fz_0=fz_0, delta_fz=delta_fz, 
                         m=m, e=e, T=T, N_average=N_average,
                         display_progress=display_progress)
        self.devices = devices
        self.tank_circuit = TankCircuit(Q, Z0, 2*np.pi*f0, T)
        
    def __call__(self, freq):
        output_ideal_spectrum = np.zeros(np.shape(freq))
        output_real_spectrum = np.zeros(np.shape(freq))

        delta_fz = self.delta_fz
        N_average = self.N_average
        n = self.n
        d = self.d
        m = self.m
        e = self.e
        delta_fz = self.delta_fz
        fz_0 = self.fz_0
        tank_circuit = self.tank_circuit

        std = delta_fz/2
        all_fz = fz_0 + np.linspace(-3*std, 3*std, N_average)
        Pz = sp.stats.norm.pdf((all_fz-fz_0)/delta_fz)
        Pz_norm = Pz / np.sum(Pz)
        
        for idx in trange(len(all_fz)):
            fz = all_fz[idx]
            electron = HarmonicElectron(n, d, 2*np.pi*fz, m, e)
            electron_trap = ElectronCoupledToTank(electron, tank_circuit)
            
            psd_input = tank_circuit.get_noise(freq)
            psd_signal = electron_trap(freq, psd_input)
            
            for device in self.devices:
                psd_signal = device(freq, psd_signal)
            
            ideal_ps, real_ps = self.measure_spectrum(freq, psd_signal)
            output_ideal_spectrum += dBm_to_W(ideal_ps) * Pz_norm[idx]
            output_real_spectrum += dBm_to_W(real_ps) * Pz_norm[idx]
        return W_to_dBm(output_ideal_spectrum), W_to_dBm(output_real_spectrum)