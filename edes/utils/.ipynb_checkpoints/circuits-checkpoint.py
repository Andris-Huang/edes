class Component:
    """
    Base class for a circuit component used.
    """
    def __init__(self, name='Device', **kwargs):
        self.name = name
        for key in kwargs:
            setattr(self, kwargs[key])
    
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
        raise NotImplemented
    
    def __call__(f, signal):
        """
        Returns the output siganl given input signal and frequencies.
        
        Parameters
        ----------
        f : ndarray
            Frequency in Hz
        signal : ndarray
            Values of the signal
        
        Returns
        -------
        V_out : ndarray
            Output signal. Default assumes linear frequency response,
            hence V_out = H(omega)V_in
        """
        omega = 2*np.pi*f
        return signal*self.H(omega)