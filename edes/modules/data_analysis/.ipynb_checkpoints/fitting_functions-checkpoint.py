import numpy as np
import scipy.constants as scc
import collections
import scipy.special as sp
import scipy.misc as scm
from scipy.optimize import curve_fit
from scipy.special import eval_genlaguerre as laguerre



class fitting_function():
    """
    This is the parent class of specific functions which are defined below.
    This class contains the function to be fitted and its parameters, including those which are to be fitted and those which are to be held at a fixed value.
    Methods are defined to set which parameters are which, set their values, and perform the fit.
    See the notebook [spacetime computer]/home/space-time/data_analysis/general_tools/fit_data.ipynb for an exmaple of use.

    Variables contained in this class:
    self.full_fun: Function. The general form of the function we want to fit to. Requires all input paramaters as arguments.
    self.parameters: List of strings. Lists the arguments that need to be passed to self.full_fun as parameters (this excludes the first argument, which is taken to be the x values.)
    self.reduced_fun: Function. The reduced form of full_fun, which only accepts the parameters that are not fixed as arguments.
    self.params_fixed: Dictionary. Keys are the names of parameters, corresponding to elements of self.parameters, and values are the values of the corresponding parameters. Only parameters not being fitted are included.
    self.params_tofit: List of strings. Subset of self.parameters; includes only those which are not keys of self.params_fixed.
    self.params_tofit_guesses: Dictionary. Keys are elements of self.params_tofit, values are guess values used as initial inputs for the actual fitting algorithm (scipy.optimize.curve_fit)
    self.params_tofit_fits: Dictionary. Keys are elements of self.params_tofit, values are 2-element tuples, with the first element being the fitted value and the second being the uncertainty (calculated as the square root of the corresponding element of the covariance matrix)
    self.fit_bounds: Dictionary. Keys are elements of self.params_tofit, values are 2-element tuples, with the first element being the lower bound and the second being the upper bound. +/-np.inf indicates no upper/lower bound.
    self.guesses_set: Boolean. Indicates whether guess parameters have been set yet. Is reset to False if self.set_fixed_params() is called.
    self.fit_done: Boolean. Indicates whether fit has been performed yet or not. Is reset to False if self.set_fixed_params() or self.set_guess_params() is called.
    """

    def __init__(self, function, parameters):
        """Fitting functions should generally be defined as child classes below.
        However, with this method you can pass in an arbitrary function and make it a fitting_function object,
        by passing in the function and a list of its parameters (following the same rules as the child classes below)."""
        self.full_fun = function
        self.parameters = parameters
        self.setup()

    def setup(self):
        self.reduced_fun = self.full_fun
        self.params_fixed = dict()
        self.params_tofit = self.parameters
        self.params_tofit_guesses = dict()
        self.params_tofit_fits = dict()
        self.fit_bounds = dict()
        self.guesses_set = False
        self.fit_done = False

    def print_parameters(self):
        print('Parameters:')
        for param in self.parameters: print("'"+param+"'")

    def set_fixed_params(self, paramstofix_dict):
        for param in paramstofix_dict.keys():
        # Make sure that the parameters we're trying to fix are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        self.guesses_set = False
        self.fit_done = False
        self.params_fixed = paramstofix_dict
        self.params_tofit = [param for param in self.parameters if param not in paramstofix_dict.keys()]
        self.params_tofit_fits = dict()
        self._set_tofit_dicts()
        self.reduced_fun = self.get_reduced_fun()

    def set_guess_params(self, paramsguess_dict):
        for param in paramsguess_dict:
            # Make sure that the parameters being set are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        for param in paramsguess_dict:
            # Make sure that none of the parameters set to be fixed are included here
            if param in self.params_fixed.keys():
                raise ValueError('"{}" is already set as a fixed parameter.'.format(param))
        for param in self.params_tofit:
            # Make sure that all parameters not being fixed are being given a guess value
            if param not in paramsguess_dict.keys():
                raise ValueError('"{0}" has not been set as a fixed parameter, but is not being given a guess value either! Give "{0}" a guess value or set it as a fixed parameter.'.format(param))
        self.guesses_set = True
        self.fit_done = False
        self.params_tofit_fits = dict()
        self.params_tofit_guesses = paramsguess_dict

    def set_fit_bounds(self, fitbounds_dict):
        for param in fitbounds_dict:
            # Make sure that the parameters being set are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        for param in fitbounds_dict:
            # Make sure that none of the parameters set to be fixed are included here
            if param in self.params_fixed.keys():
                raise ValueError('"{}" is already set as a fixed parameter.'.format(param))
        # for param in fitbounds_dict:
        #     # Make sure the value being passed for each parameter is actually a 2-tuple
        #     bounds = fitbounds_dict[param]
        #     assert isinstance(bounds, collections.Sequence), 'Bounds must be set as a 2-tuple. Use (+ or -) np.inf to indicate no (upper or lower) bound.'
        #     assert len(bounds) == 2,                         'Bounds must be set as a 2-tuple. Use (+ or -) np.inf to indicate no (upper or lower) bound.'
        self.fit_done = False
        self.params_tofit_fits = dict()
        self.fit_bounds = fitbounds_dict

    def do_fit(self, x, y, yerr=None, use_qpn=False, nmeasurements=100, print_result=False):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call fitting_function.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        # Construct the keyword argument inputs for p0 and bounds
        p0 = self._guess_params_to_args()
        bounds = self._bounds_to_lists()
        
        # If use_qpn is true, the quantum-projection-noise-calculated uncertainty is used.
        # For excitation p from N measurements, this is given by either sqrt(p*(1-p)/N) or 1/(N+2), whichever is larger [Thomas Monz thesis, Innsbruck]
        if use_qpn==True:
            yerr = np.maximum(np.sqrt(y*(1.0-y)/nmeasurements), 1.0/(nmeasurements+2))
        elif use_qpn=='parity':
            p = (y+1)/2.0
            yerr = np.maximum(np.sqrt(p*(1.0-p)/nmeasurements), 2.0/(nmeasurements+2))
        
        # absolute_sigma should be used in curve_fit only if a y error has been specified.
        abs_sig = True if yerr is not None else False

        # Get the fitted values and uncertainties
        (fitvals, covar) = curve_fit(self.reduced_fun, x, y, sigma=yerr, p0=p0, bounds=bounds, absolute_sigma=abs_sig, xtol=1e-10, ftol=1e-10, maxfev=10000)
        
        # Write the fitted values and uncertainties to the dictionary self.params_tofit_fits
        for (i, param) in enumerate(self.params_tofit):
            self.params_tofit_fits[param] = (fitvals[i], np.sqrt(covar[i, i]))
        
        # Write residuals and rsq to self.residuals and self.rsq
        self.residuals = np.array(y)-self.reduced_fun(np.array(x),*fitvals)
        ss_r = np.sum(self.residuals**2)
        ss_tot = np.sum((np.array(y)-np.mean(y))**2)
        self.rsq = 1-(ss_r/ss_tot)
        
        self.fit_done = True
        if print_result:
            self.print_fits()
    
    def eval_with_guesses(self, x):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call fitting_function.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        reduced_args = self._guess_params_to_args()
        return self.reduced_fun(x, *reduced_args)

    def eval_with_fits(self, x):
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call fitting_function.do_fit().')
        reduced_args = self._fit_params_to_args()
        return self.reduced_fun(x, *reduced_args)
    
    def eval_with_bounds(self,x):
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call fitting_function.do_fit().')
        reduced_argsT = [None]*len(self.params_tofit)
        reduced_argsB = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            reduced_argsT[i] = self.params_tofit_fits[param][0]+self.params_tofit_fits[param][1]
            reduced_argsB[i] = self.params_tofit_fits[param][0]-self.params_tofit_fits[param][1]
        return self.reduced_fun(x,*reduced_argsT), self.reduced_fun(x,*reduced_argsB)
    
    def get_status(self):
        return {'guesses_set':self.guesses_set, 'fit_done':self.fit_done}
    
    def get_fixedparams(self):
        return self.params_fixed.copy()

    def get_guesses(self):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call fitting_function.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        return self.params_tofit_guesses.copy()
    
    def get_fits(self):
        # Returns self.params_tofit_fits, a dictionary of fitted values and uncertainties
        # Each key is the name of a fitted parameter, and each value is a 2-tuple of the format (fitvalue, uncertainty)
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call fitting_function.do_fit().')
        return self.params_tofit_fits.copy()

    def get_reduced_fun(self):
        # Returns a function that accepts a reduced set of input arguments, i.e. with the fixed parameters set to their specified values (as defined by the dictionary self.params_fixed)
        def reduced_fun(x, *reduced_args):
            reduced_args = list(reduced_args)
            full_args = [None]*len(self.parameters)
            for (i, param) in enumerate(self.parameters):
                if param in self.params_tofit:
                    full_args[i] = reduced_args.pop(0)
                else:
                    full_args[i] = self.params_fixed[param]
            return self.full_fun(x, *full_args)
        return reduced_fun

    def print_fits(self):
        # Print out the fitted values and uncertainties
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call fitting_function.do_fit().')
        for param in self.params_tofit_fits:
            print('{0}: {1} +- {2}'.format(param, self.params_tofit_fits[param][0], self.params_tofit_fits[param][1]))

    def _set_tofit_dicts(self):
        # Called after fixed parameters have been set.
        # Initializes the dictionaries params_tofit_guesses and params_tofit_fits with the correct keys (i.e. the parameters to be fitted), and None values.
        params_tofit_guesses = dict()
        params_tofit_fits = dict()
        for param in self.params_tofit:
            params_tofit_guesses[param] = None
            params_tofit_fits[param] = None

    def _guess_params_to_args(self):
        # Takes the information stored in params_tofit_guesses and turns it into a list of fit values ordered by the relevent parameters' ordering in self.parameters
        reduced_args = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            reduced_args[i] = self.params_tofit_guesses[param]
        return reduced_args

    def _fit_params_to_args(self):
        # Takes the information stored in params_tofit_fits and turns it into a list of fit values ordered by the relevent parameters' ordering in self.parameters
        reduced_args = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            reduced_args[i] = self.params_tofit_fits[param][0]
        return reduced_args

    def _bounds_to_lists(self):
        # Turns the fitting bounds set in the dictionary self.fit_bounds into a form which can be passed into scipy.optimize.curve_fit's "bounds" keyword argument
        lbounds = [None]*len(self.params_tofit)
        ubounds = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            if param in self.fit_bounds.keys():
                lbounds[i] = self.fit_bounds[param][0]
                ubounds[i] = self.fit_bounds[param][1]
            else:
                lbounds[i] = -np.inf
                ubounds[i] = np.inf
        return (lbounds, ubounds)






def print_functions():
    # Print out a list of the fitting functions which have been defined
    import sys, inspect
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    print('Options:')
    for (name, obj) in clsmembers:
        if name != 'fitting_function':
            print(name+'()')

#############################################################################################################################################################################################

"""
Below are the functions which have been defined, as child classes of the class fitting_function.

Defining a new fitting function is simple! (I hope.)

Short version:
Just copy one of the classes already defined. It may be clear what needs to be changed given the examples below. If not...
Only change (1) self.parameters and (2) the definition of full_fun() (which is the function you want to fit to).
The first argument to full_fun() after "self" should be the x-axis values. Make sure the elements of the list self.parameters match all the arguments after this one.

Long version:
To define a function to be used as a fitting function, define the class as follows:
A. Define the class as a child instance of the class fitting_function.
B. __init__ should consist of 2 lines:
   1. self.parameters = (a list of strings, see step C for details)
   2. self.setup()
C. Define a function full_fun(self, ...) to be the function to be fitted. The first argument after self should be the x-axis, and the rest are just any arguments the function needs.
   These will each later be set to either be fitted or fixed at some defined value.
   The variable self.parameters should be a list of strings denoting these parameters (the ones excluding the x-axis), so the length of this list should be equal to the number of arguments
   excluding the first one. These don't necessarily need to have the same name as the arguments themselves (though it's probably easier that way), but the arguments they refer to *DO* need to
   be in the same order.
"""


class linear(fitting_function):
    def __init__(self):
        self.parameters = ['m', 'b']
        self.setup()
        
    def full_fun(self, x, m, b):
        return m*x + b

    
class exponential_decay(fitting_function):
    def __init__(self):
        self.parameters = ['tau', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, tau, startfrom, decayto):
        return startfrom + (decayto-startfrom)*(1-np.exp(-times/tau))

class double_exponential(fitting_function):
    def __init__(self):
        self.parameters = ['tau1', 'startfrom1', 'decayto1', 'tau2', 'startfrom2', 'decayto2']
        self.setup()

    def full_fun(self, times, tau1, startfrom1, decayto1, tau2, startfrom2, decayto2):
        return startfrom1 + (decayto1-startfrom1)*(1-np.exp(-times/tau1)) + startfrom2 + (decayto2-startfrom2)*(1-np.exp(-times/tau2))

    
class gaussian_decay(fitting_function):
    def __init__(self):
        self.parameters = ['tau', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, tau, startfrom, decayto):
        return startfrom + (decayto-startfrom)*(1-np.exp(-times**2/(2*tau**2)))
    
    
class cpmg_decay1(fitting_function):
    #Following Wang et al PRB 2012 "Comparison of Dynamical Decoupling Protocols for a NV Center in Diamond"
    def __init__(self):
        self.parameters = ['T2','Nd', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, T2, Nd, startfrom, decayto):
        tau = T2*(Nd/2.0)**(2.0/3.0)
        return startfrom + (decayto-startfrom)*(1-np.exp(-(times**3)/(tau**3)))

    
class cpmg_decay2(fitting_function):
    # Same as cpmg_decay1 except combining T2 and Nd into a single time decay constant tau
    def __init__(self):
        self.parameters = ['tau', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, tau, startfrom, decayto):
        return startfrom + (decayto-startfrom)*(1-np.exp(-(times**3)/(tau**3)))


class cpmg_decay2_wcontrastosc(fitting_function):
    # Same as cpmg_decay2 but with contrast oscillations
    # These depend on the value of Delta_l and the rotor constant (calculated from f_trap_MHz and f_rot_kHz)
    # The parameters a and b are empirical values that account for imperfect operations. Can use simulations/rotating_interferometer/simulations_of_expt/rotor_sim_rabi_and_ramsey.ipynb to get them.
    # See also lablog/spacetime/?q=node/625

    def __init__(self):
        self.parameters = ['tau_ms', 'Delta_l', 'f_trap_MHz', 'f_rot_kHz', 'a', 'b', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times_ms, tau_ms, Delta_l, f_trap_MHz, f_rot_kHz, a, b, startfrom, decayto):
        t = 1e-3 * times_ms 

        w_trap = 1e6 * 2*np.pi * f_trap_MHz
        w_rot = 1e3 * 2*np.pi * f_rot_kHz

        # calculate moment of inertia
        m = 40*scc.atomic_mass
        r0 = 1/2.0 * (scc.e**2/(4*np.pi*scc.epsilon_0) * 2.0/(m*w_trap**2))**(1.0/3) # undistorted rotor radius
        I0 = 2*m*r0**2  # undistorted moment of inertia

        # calculate effective rotor frequency
        w_r = scc.hbar/(2*I0)                                    # Non-distorted rotor frequency
        D = 4*w_r**3 / (3*w_trap**2)                             # First-order distortion constant
        l_0 = min(np.abs(np.roots([4*D, 0, -2*w_r, w_rot])))     # Angular momentum expectation value with first-order distortion (solves equation for l_0 given w_rot which is cubic in l_0)
        w_r_eff = w_r - 6*D*l_0**2                               # "Effective" in terms of spacing of the transition frequencies

        contrast_osc_envelope = np.abs( (a+b) + (a-b)*np.cos(Delta_l**2*w_r_eff*t) - 1 ) / (2*a - 1)
        decoherence_envelope = np.exp(-(times_ms**3)/(tau_ms**3))
        return startfrom + (decayto-startfrom)*(1 - contrast_osc_envelope*decoherence_envelope)


class cosine(fitting_function):
    def __init__(self):
        self.parameters = ['angfreq', 'amplitude', 'offset', 'phase_deg']
        self.setup()

    def full_fun(self, x, angfreq, amplitude, offset, phase_deg):
        return offset + amplitude*np.cos(angfreq*x + np.pi/180*phase_deg)
    

class quadrupole_scan(fitting_function):
    def __init__(self):
        self.parameters = ['U1', 'U3', 'offset']
        self.setup()

    def full_fun(self, x, U1, U3, offset):
        return offset + U1*np.cos(np.pi/180*(x - 300)) + U3*np.cos(np.pi/180*(x - 210))

class lorentzian(fitting_function):
    #defined as in grapher
    def __init__(self):
        self.parameters = ['center', 'scale', 'fwhm', 'offset']
        self.setup()
        
    def full_fun(self, frequency, center, scale, fwhm, offset):
        return offset + scale*0.5*fwhm/((frequency-center)**2.0 + fwhm**2.0/4.0)
    
    
class gaussian(fitting_function):
    #defined as in grapher
    def __init__(self):
        self.parameters = ['center', 'scale', 'sigma', 'offset']
        self.setup()
        
    def full_fun(self, frequency, center, scale, sigma, offset):
        return offset + scale*np.exp(-(frequency-center)**2 / (2*sigma**2))

class double_gaussian(fitting_function):
    # defined as in grapher
    # assumes same standard deviation for both, and a single global offset
    def __init__(self):
        self.parameters = ['center1', 'scale1', 'center2', 'scale2', 'sigma', 'offset']
        self.setup()
        
    def full_fun(self, frequency, center1, scale1, center2, scale2, sigma, offset):
        return offset + scale1*np.exp(-(frequency-center1)**2 / (2*sigma**2)) + scale2*np.exp(-(frequency-center2)**2 / (2*sigma**2))
    

class poissonian(fitting_function):
    def __init__(self):
        # mu is center, k is value, k has to be positive
        self.parameters = ['mu']
        self.setup()
    
    def full_fun(self,k,mu):
        return np.exp(-mu)*mu**k/scm.factorial(k)

class poissonian2(fitting_function):
    # two poisson mass functions for fitting to readout histograms. 
    # (assuming no relevant  D state decay)
    def __init__(self):
        # mu is center, k is value, k has to be positive
        self.parameters = ['mu1','mu2']
        self.setup()
    
    def full_fun(self,k,mu1,mu2):
        return np.exp(-mu1)*mu1**k/scm.factorial(k) + np.exp(-mu2)*mu2**k/scm.factorial(k)
    
class phase_scan(fitting_function):
    def __init__(self):
        self.parameters = ['contrast', 'phi0', 'offset']
        self.setup()

    def full_fun(self, phi, contrast, phi0, offset):
        phi_rad = phi*np.pi/180.0
        phi0_rad = phi0*np.pi/180.0
        return 0.5 + contrast/2.0*np.sin(phi_rad-phi0_rad) + offset
    
    
class parity_scan(fitting_function):
    def __init__(self):
        self.parameters = ['contrast', 'phi0', 'offset']
        self.setup()

    def full_fun(self, phi, contrast, phi0, offset):
        phi_rad = phi*np.pi/180.0
        phi0_rad = phi0*np.pi/180.0
        return contrast/2.0*np.sin(2*(phi_rad-phi0_rad)) + offset
    
    
class ramsey_decay(fitting_function):
    def __init__(self):
        self.parameters = ['freq', 'tau_us', 'start_from', 'decay_to']
        self.setup()
        
    def full_fun(self, times_us, freq, tau_us, start_from, decay_to):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 2*np.pi*freq
        tau = 1e-6 * tau_us
        
        return (start_from-decay_to)*np.exp(-times/tau)*np.cos(w*times) + decay_to

class ramsey_AC_field_DD(fitting_function):
    def __init__(self):
        self.parameters = ['freq_kHz', 'tau_us', 'start_from', 'offset', 'B_kHz', 'phase_deg']
        self.setup()
        
    def full_fun(self, times_us, freq_kHz, tau_us, start_from, offset, B_kHz, phase_deg):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 1e3 * 2*np.pi*freq_kHz
        tau = 1e-6 * tau_us
        phase = np.pi*phase_deg/180
        amp = 1e3 * 2*np.pi*B_kHz
        
        theta = amp*(np.cos(w*times+phase) + np.cos(phase) - 2*np.cos(w*times/2+phase))/w
        
        return start_from*np.exp(-times/tau)*((1-np.cos(theta))/2-0.5) + 0.5 + offset

    
class ramsey_AC_field_DD_incoherent(fitting_function):
    def __init__(self):
        self.parameters = ['freq', 'tau_us', 'start_from', 'offset', 'B_kHz']
        self.setup()
        
    def full_fun(self, times_us, freq, tau_us, start_from, offset, B_kHz):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 2*np.pi*freq
        tau = 1e-6 * tau_us
        amp = 1e3 * 2*np.pi*B_kHz
        
        D = 4*amp*(np.sin(w*times/4))**2/w
        
        return start_from*np.exp(-times/tau)*((1-sp.jv(0,D))/2-0.5) + 0.5 + offset


class ramsey_AC_field(fitting_function):
    def __init__(self):
        self.parameters = ['freq_kHz', 'B_ac_kHz', 'B_dc_kHz', 'phase_sig_deg', 'phase_pi2_deg']
        self.setup()
        
    def full_fun(self, times_us, freq_kHz, B_ac_kHz, B_dc_kHz, phase_sig_deg, phase_pi2_deg):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 1e3 * 2*np.pi*freq_kHz
        B1 = 1e3 * 2*np.pi*B_ac_kHz
        B0 = 1e3 * 2*np.pi*B_dc_kHz
        phase_sig = np.pi/180 * phase_sig_deg
        phase_pi2 = np.pi/180 * phase_pi2_deg
                
        return (1 + np.cos(B1/w*(np.sin(w*times+phase_sig)-np.sin(phase_sig)) + B0*times - phase_pi2))/2


class ramsey_AC_field_incoherent(fitting_function):
    def __init__(self):
        self.parameters = ['freq_kHz', 'B_ac_kHz', 'B_dc_kHz', 'phase_pi2']
        self.setup()
        
    def full_fun(self, times_us, freq_kHz, B_ac_kHz, B_dc_kHz, phase_pi2_deg):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 1e3 * 2*np.pi*freq_kHz
        B1 = 1e3 * 2*np.pi*B_ac_kHz
        B0 = 1e3 * 2*np.pi*B_dc_kHz
        phase = np.pi/180 * phase_pi2_deg
                
        return (1 + sp.jv(0, 2*B1/w*np.sin(w*times/2))*np.cos(B0*times-phase))/2
    
class gEst(fitting_function):
    def __init__(self):
        self.parameters = ['gmin']
        self.setup()
        
    def full_fun(self, gs, gmin):
        return (gmin**4. + gs**4.)**0.25
    
class gEst2(fitting_function):
    def __init__(self):
        self.parameters = ['gmin']
        self.setup()
        
    def full_fun(self, gs, gmin):
        gmeas = []
        for g in gs:
            if g < gmin:
                gmeas.append(gmin)
            else:
                gmeas.append(g)
        
        return np.array(gmeas)


class rot_ramsey_decay(fitting_function):
    def __init__(self):
        self.parameters = ['sigma_l', 'Omega_kHz', 'delta_kHz', 'scale', 'Delta_l', 'f_trap_MHz', 'f_rot_kHz']
        self.setup()

    def full_fun(self, times_us, sigma_l, Omega_kHz, delta_kHz, scale, Delta_l, f_trap_MHz, f_rot_kHz):
        # convert inputs to SI
        times = 1e-6 * times_us
        Omega = 1e3 * 2*np.pi * Omega_kHz
        delta = 1e3 * 2*np.pi * delta_kHz
        w_trap = 1e6 * 2*np.pi * f_trap_MHz
        w_rot = 1e3 * 2*np.pi * f_rot_kHz

        # calculate moment of inertia
        m = 40*scc.atomic_mass
        r0 = 1/2.0 * (scc.e**2/(4*np.pi*scc.epsilon_0) * 2.0/(m*w_trap**2))**(1.0/3) # undistorted rotor radius
        I0 = 2*m*r0**2  # undistorted moment of inertia

        # calculate effective rotor frequency
        w_r = scc.hbar/(2*I0)                                    # Non-distorted rotor frequency
        D = 4*w_r**3 / (3*w_trap**2)                             # First-order distortion constant
        l_0 = min(np.abs(np.roots([4*D, 0, -2*w_r, w_rot])))     # Angular momentum expectation value with first-order distortion (solves equation for l_0 given w_rot which is cubic in l_0)
        w_r_eff = w_r - 6*D*l_0**2                               # "Effective" in terms of spacing of the transition frequencies

        # calculate l distribution and detunings
        ls = np.arange(int(l_0-3*sigma_l), int(l_0+3*sigma_l))
        c_ls_unnorm = np.exp(-(ls-l_0)**2/(4.0*sigma_l**2))
        c_ls = c_ls_unnorm/np.linalg.norm(c_ls_unnorm)
        delta_ls = 2*w_r_eff*Delta_l*(l_0-ls) + delta

        def calc_ramsey_exc(c_ls, delta_ls, Omega, T):
            Omega_gens = np.sqrt(Omega**2 + delta_ls**2) #generalized Rabi frequency
            u1s = np.pi*Omega_gens/(4*Omega)
            u2s = delta_ls*T/2.0
            return sum(np.abs(c_ls)**2 * (2*Omega/Omega_gens**2*np.sin(u1s) * (Omega_gens*np.cos(u1s)*np.cos(u2s) - delta_ls*np.sin(u1s)*np.sin(u2s)))**2)
            
        return [scale * calc_ramsey_exc(c_ls, delta_ls, Omega, T) for T in times]


class rot_ramsey_decay_general(fitting_function):
    def __init__(self):
        self.parameters = ['sigma_l', 'delta_kHz', 'Delta_l', 'f_trap_MHz', 'f_rot_kHz', 'scale', 'contrast', 'phase']
        self.setup()

    def full_fun(self, times_us, sigma_l, delta_kHz, Delta_l, f_trap_MHz, f_rot_kHz, scale, contrast, phase):
        # convert inputs to SI
        times = 1e-6 * times_us
        delta = 1e3 * 2*np.pi * delta_kHz
        w_trap = 1e6 * 2*np.pi * f_trap_MHz
        w_rot = 1e3 * 2*np.pi * f_rot_kHz

        # calculate moment of inertia
        m = 40*scc.atomic_mass
        r0 = 1/2.0 * (scc.e**2/(4*np.pi*scc.epsilon_0) * 2.0/(m*w_trap**2))**(1.0/3) # undistorted rotor radius
        I0 = 2*m*r0**2  # undistorted moment of inertia

        # calculate effective rotor frequency
        w_r = scc.hbar/(2*I0)                                    # Non-distorted rotor frequency
        D = 4*w_r**3 / (3*w_trap**2)                             # First-order distortion constant
        l_0 = min(np.abs(np.roots([4*D, 0, -2*w_r, w_rot])))     # Angular momentum expectation value with first-order distortion (solves equation for l_0 given w_rot which is cubic in l_0)
        w_r_eff = w_r - 6*D*l_0**2                               # "Effective" in terms of spacing of the transition frequencies
        
        sigma_f = 2*w_r_eff*Delta_l*sigma_l               # Frequency-space standard deviation of the line
        Ct = contrast*np.exp(-(sigma_f*times)**2/2.0)     # Inverse Fourier transform of Gaussian lineshape from frequency domain to time domain
        return scale * (Ct*np.cos(delta*times-phase) + 1.0)/2.0


class rot_rabi_flop(fitting_function):
    def __init__(self):
        self.parameters = ['sigma_l', 'Omega_kHz', 'delta_kHz', 'Delta_l', 'f_trap_MHz', 'f_rot_kHz', 'scale']
        self.setup()

    def full_fun(self, times_us, sigma_l, Omega_kHz, delta_kHz, Delta_l, f_trap_MHz, f_rot_kHz, scale):
        if sigma_l > 3000:
            sigma_l = 3000.0

        times = 1e-6 * times_us
        Omega = 1e3 * 2*np.pi * Omega_kHz
        delta = 1e3 * 2*np.pi * delta_kHz
        w_trap = 1e6 * 2*np.pi * f_trap_MHz
        w_rot = 1e3 * 2*np.pi * f_rot_kHz

        # calculate moment of inertia
        m = 40*scc.atomic_mass
        r0 = 1/2.0 * (scc.e**2/(4*np.pi*scc.epsilon_0) * 2.0/(m*w_trap**2))**(1.0/3) # undistorted rotor radius
        I0 = 2*m*r0**2  # undistorted moment of inertia

        # calculate effective rotor frequency
        w_r = scc.hbar/(2*I0)                                    # Non-distorted rotor frequency
        D = 4*w_r**3 / (3*w_trap**2)                             # First-order distortion constant
        l_0 = min(np.abs(np.roots([4*D, 0, -2*w_r, w_rot])))     # Angular momentum expectation value with first-order distortion (solves equation for l_0 given w_rot which is cubic in l_0)
        w_r_eff = w_r - 6*D*l_0**2                               # "Effective" in terms of spacing of the transition frequencies

        # calculate l distribution and detunings
        ls = np.arange(int(l_0-3*sigma_l), int(l_0+3*sigma_l))
        c_ls_unnorm = np.exp(-(ls-l_0)**2/(4.0*sigma_l**2))
        c_ls = c_ls_unnorm/np.linalg.norm(c_ls_unnorm)
        delta_ls = 2*w_r_eff*Delta_l*(l_0-ls) + delta

        exc = scale * np.sum(np.outer(c_ls**2 * Omega**2/(Omega**2+delta_ls**2), np.ones(len(times))) \
                    * np.sin(np.outer(np.sqrt(Omega**2+delta_ls**2)/2, times))**2, axis=0)

        return exc


class rot_rabi_flop_with_finite_nbar(fitting_function):
    # This is a little against the spirit of this suite in that it differs from rot_rabi_flop() by only having 2 extra parameters (nzbar and f_z_MHz, used to calculate a Lamb-Dicke parameter for the vertical direction).
    # The point of this suite generally is to have a class that defined a type of function very generally, and then use the built-in stuff to de-generalize it as much as you need.
    # This suggests that since rot_rabi_flop_with_finite_nbar() is more general than rot_rabi_flop(), rot_rabi_flop() is then redundant.
    # HOWEVER. I think that the extra parameters in rot_rabi_flop_with_finite_nbar() are sufficiently different/specific that it merits its own class, in this particular case.
    def __init__(self):
        self.parameters = ['sigma_l', 'Omega_kHz', 'delta_kHz', 'Delta_l', 'f_trap_MHz', 'f_rot_kHz', 'nzbar', 'f_z_MHz', 'scale']
        self.setup()

    def full_fun(self, times_us, sigma_l, Omega_kHz, delta_kHz, Delta_l, f_trap_MHz, f_rot_kHz, nzbar, f_z_MHz, scale):
        if sigma_l > 3000:
            sigma_l = 3000.0

        times = 1e-6 * times_us
        Omega = 1e3 * 2*np.pi * Omega_kHz
        delta = 1e3 * 2*np.pi * delta_kHz
        w_trap = 1e6 * 2*np.pi * f_trap_MHz
        w_rot = 1e3 * 2*np.pi * f_rot_kHz

        # calculate moment of inertia
        m = 40*scc.atomic_mass
        r = 1/2.0 * (scc.e**2/(4*np.pi*scc.epsilon_0) * 2.0/(m*(w_trap**2 - w_rot**2)))**(1/3.0) #rotor radius
        I = 2*m*r**2  # moment of inertia

        # calculate l distribution and detunings
        l_0 = I*w_rot/scc.hbar
        ls = np.arange(int(l_0-3*sigma_l), int(l_0+3*sigma_l))
        c_ls_unnorm = np.exp(-(ls-l_0)**2/(4.0*sigma_l**2))
        c_ls = c_ls_unnorm/np.linalg.norm(c_ls_unnorm)
        delta_ls = scc.hbar*Delta_l/I*(l_0-ls) + delta

        if nzbar == 0:
            exc = scale * np.sum(np.outer(c_ls**2 * Omega**2/(Omega**2+delta_ls**2), np.ones(len(times))) \
                    * np.sin(np.outer(np.sqrt(Omega**2+delta_ls**2)/2, times))**2, axis=0)
        else:
            w_z = 1e6 * 2*np.pi * f_z_MHz
            eta = 2*np.pi/(729e-9)*np.sqrt(scc.hbar/(2*2*m*w_z))

            Omega_gens = np.sqrt(Omega**2 + delta_ls**2) #generalized Rabi frequency
            phases = np.outer(Omega_gens, times) #a 2d array of values of Omega_gen*t, defined here for convenience

            exc = scale * np.sum(np.outer(c_ls**2 * Omega**2/Omega_gens**2, np.ones(len(times))) \
                    * 1/2.0*(1 - (np.cos(phases) + phases*eta**2*nzbar*np.sin(phases))/(1 + (phases*eta**2*nzbar)**2)), axis=0)
        return exc


class rabi_flop_thermal(fitting_function):
    # Supports up to second order sidebands
    def __init__(self):
        self.parameters = ['Omega_kHz', 'delta_kHz', 'f_trap_MHz', 'sideband_order', 'scale', 'n_ions', 'nbar', 'turnon_delay_us']
        self.setup()

    def full_fun(self, times_us, Omega_kHz, delta_kHz, f_trap_MHz, sideband_order, scale, n_ions, nbar, turnon_delay_us):
        m = 40*scc.atomic_mass

        times = 1e-6 * np.array([(t if t>=0 else 0) for t in (times_us-turnon_delay_us)])
        Omega = 1e3 * 2*np.pi * Omega_kHz
        delta = 1e3 * 2*np.pi * delta_kHz
        w_trap = 1e6 * 2*np.pi * f_trap_MHz


        eta = 2*np.pi/(729e-9)*np.sqrt(scc.hbar/(2*m*n_ions*w_trap))

        nmax = 1000
        ns = np.arange(nmax)

        omega_n = Omega*self._compute_rabi_coupling(eta, sideband_order, ns)
        p_n = 1.0/ (nbar + 1.0) * (nbar / (nbar + 1.0))**ns
        exc_n = np.outer(p_n * omega_n**2/(omega_n**2+delta**2), np.ones_like(times)) * np.sin(np.outer(np.sqrt(omega_n**2 + delta**2), times/2.0))**2
        exc = scale * np.sum(exc_n, axis = 0)

        return exc

    def _compute_rabi_coupling(self, eta, sideband_order, ns):
        if sideband_order == 0:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * laguerre(n, 0, eta**2)
        elif sideband_order == 1:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * eta**(1)*(1./(n+1.))**0.5 * laguerre(n, 1, eta**2)
        elif sideband_order == 2:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * eta**(2)*(1./((n+1.)*(n+2)))**0.5 * laguerre(n, 2, eta**2)
        elif sideband_order == -1:
            coupling_func = lambda n: 0 if n == 0 else np.exp(-1./2*eta**2) * eta**(1)*(1./(n))**0.5 * laguerre(n - 1, 1, eta**2)
        elif sideband_order == -2:
            coupling_func = lambda n: 0 if n <= 1 else np.exp(-1./2*eta**2) * eta**(2)*(1./((n)*(n-1.)))**0.5 * laguerre(n - 2, 2, eta**2)
        return np.array([coupling_func(n) for n in ns])


class power_law(fitting_function):
    def __init__(self):
        self.parameters = ['a', 'b']
        self.setup()
        
    def full_fun(self, x, a, b):
        return a*x**b
    
    
class diffusion(fitting_function):
    def __init__(self):
        self.parameters = ['D', 'y0']
        self.setup()
        
    def full_fun(self, t, D, y0):
        return np.sqrt(2*D*(t+y0**2/(2*D)))
    

class frequency_resolution_DD(fitting_function):
    def __init__(self):
        self.parameters = ['B','T','offset']
        self.setup()
    
    def full_fun(self,wr, B, T,offset):
        B = 2*np.pi*B
        wr = wr*2*np.pi
        return (wr**4)*(B**2)*(T**6)/((8**4)*((2*np.pi)**2)) + offset
    

class frequency_resolution(fitting_function):
    def __init__(self):
        self.parameters = ['B','T','offset']
        self.setup()
    
    def full_fun(self,wr, B, T,offset):
        B = 2*np.pi*B
        wr = wr*2*np.pi
        return (wr**2)*(B**2)*(T**4)/(16*(np.pi**2)) + offset


class spectrum_2level(fitting_function):
    # Spectrum of an ideal 2-level transition
    def __init__(self):
        self.parameters = ['f0_MHz', 'Omega_kHz', 'time_us', 'scale']
        self.setup()
        
    def full_fun(self, f_MHz, f0_MHz, Omega_kHz, time_us, scale):
        delta = 1e6 * 2*np.pi * (f_MHz-f0_MHz)
        Omega = 1e3 * 2*np.pi * Omega_kHz
        time = 1e-6 * time_us
        return scale * Omega**2/(Omega**2+delta**2) * np.sin(np.sqrt(Omega**2+delta**2)*time/2)**2


class spectrum(fitting_function):
    # Spectrum whose value at delta = 0 is not defined by the value of sin^2(Omega*t/2) but rather directly by the parameter 'scale'
    def __init__(self):
        self.parameters = ['f0_MHz', 'Omega_kHz', 'scale']
        self.setup()
        
    def full_fun(self, f, f0_MHz, Omega_kHz, scale):
        delta = 1e6 * 2*np.pi * (f-f0_MHz)
        Omega = 1e3 * 2*np.pi * Omega_kHz
        time = np.pi / Omega
        return scale * Omega**2/(Omega**2+delta**2) * np.sin(np.sqrt(Omega**2+delta**2)*time/2)**2


class sinc_squared(fitting_function):
    def __init__(self):
        self.parameters = ['x0', 'width', 'scale']
        self.setup()
        
    def full_fun(self, x, x0, width, scale):
        return scale*np.sinc((x-x0)/width)**2


class rot_ramsey_decay_contrast(fitting_function):
    def __init__(self):
        self.parameters = ['omega_r', 'Delta_l', 'sigma_l', 'contrast']
        self.setup()
    def full_fun(self, times_ms, omega_r, Delta_l, sigma_l, contrast):
        omega_r_kHz = omega_r*1e-3
        return  contrast*np.exp(-(2*omega_r_kHz*sigma_l*Delta_l)**2 * (times_ms)**2)


class rot_ramsey_decay_contrast_echo(fitting_function):
    def __init__(self):
        self.parameters = ['D', 'omega_r', 'Delta_l', 'scale']
        self.setup()
    def full_fun(self, times_ms, D, omega_r, Delta_l, scale):
        return  scale*np.exp(-D*times_ms/2.0 * (1-np.sin(2*omega_r*Delta_l*times_ms*1e-3)/(2*omega_r*Delta_l*times_ms*1e-3)))
    
class rot_ramsey_decay_echo_contrast_oscillation(fitting_function):
    def __init__(self):
        self.parameters = ['D', 'omega_r', 'Delta_l', 'a', 'b', 'T_electronic_ms', 'T_motional_ms']
        self.setup()
    def full_fun(self, times_ms, D, omega_r, Delta_l, a, b, T_electronic_ms, T_motional_ms):
        C_oscillation = np.abs((a + b) + (a - b) * np.cos(Delta_l**2 * omega_r * times_ms*1e-3) - 1)
        C_electronic = np.exp(-times_ms/T_electronic_ms)
        C_motional = np.exp(-times_ms/T_motional_ms)
        return  C_motional*C_electronic*C_oscillation*np.exp(-D*times_ms/2.0 * (1-np.sin(2*omega_r*Delta_l*times_ms*1e-3)/(2*omega_r*Delta_l*times_ms*1e-3)))
class exchange_contrast(fitting_function):
    def __init__(self):
        self.parameters = ['T_rev', 'omega_r', 'Delta_l', 'sigma_l', 'contrast']
        self.setup()
    def full_fun(self, times_ms, T_rev, omega_r, Delta_l, sigma_l, contrast):
        omega_r_kHz = omega_r*1e-3
        return  contrast*np.exp(-(2*omega_r_kHz*sigma_l*Delta_l)**2 * (times_ms - T_rev)**2)