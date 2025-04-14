import numpy as np
import matplotlib.pyplot as plt
import torch


class actionAngleHarmonic():
    """Action-angle formalism for the one-dimensional harmonic oscillator"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleHarmonic object.

        Parameters
        ----------
        omega : float or numpy.ndarray
            Frequencies (can be Quantity).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-04-08 - Written - Bovy (Uoft)

        """
        if not "omega" in kwargs:  # pragma: no cover
            raise OSError("Must specify omega= for actionAngleHarmonic")
        omega = kwargs.get("omega")
        self._omega = omega
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the action for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        float or numpy.ndarray
            action

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the action and frequency for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            (action,frequency)

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (
                (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0,
                self._omega * torch.ones_like(x),
            )
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the action, frequency, and angle for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            (action,frequency,angle)

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (
                (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0,
                self._omega * torch.ones_like(x),
                torch.arctan2(self._omega * x, vx),
            )
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")


class actionAngleHarmonicInverse():
    def __init__(self, *args, **kwargs):
        '''
        Inverse action-angle transformation for the one-dimensional 
        harmonic oscillator written in pytorch

        Parameters:
        -----------
        omega : float
            Frequency of the harmonic oscillator
        '''
        #actionAngleInverse.__init__(self, *args, **kwargs)
        if not "omega" in kwargs:  # pragma: no cover
            raise OSError("Must specify omega= for actionAngleHarmonic")
        omega = kwargs.get("omega")
        self._omega = omega
        return None
    
    def __call__(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle

        Returns
        -------
        x_vx : list
            A list containing the phase-space coordinates [x,vx]
        """
        try:
            return self.evaluate(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'__call__' method not implemented for this actionAngle module"
            )
    
    def evaluate(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle

        Returns
        -------
        x_vx : list
            A list containing the phase-space coordinates [x,vx]

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        return self.xvFreqs(j, angle, **kwargs)

    def xvFreqs(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency

        Parameters
        ----------
        j : float
            Action.
        angle : numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            Tuple containing:
                - x (numpy.ndarray): x-coordinate.
                - vx (numpy.ndarray): Velocity in x-direction.
                - Omega (float): Frequency.

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        amp = torch.sqrt(2.0 * j / self._omega)
        x = amp * torch.sin(angle)
        vx = amp * self._omega * torch.cos(angle)
        return torch.cat((x[None], vx[None]))

    def Freqs(self, j, **kwargs):
        """
        Return the frequency corresponding to a torus

        Parameters
        ----------
        j : scalar
            action

        Returns
        -------
        Omega : float
            frequency

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        return self._omega
    

class actionAngleHarmonicInverse2D():
    def __init__(self, *args, **kwargs):
        '''
        Inverse action-angle transformation for the one-dimensional 
        harmonic oscillator written in pytorch

        Parameters:
        -----------
        omega : float
            Frequency of the harmonic oscillator
        '''
        #actionAngleInverse.__init__(self, *args, **kwargs)
        if not "omega" in kwargs:  # pragma: no cover
            raise OSError("Must specify omega= for actionAngleHarmonic")
        omega = kwargs.get("omega")
        self._omega = omega
        return None
    
    def __call__(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle

        Returns
        -------
        x_vx : list
            A list containing the phase-space coordinates [x,vx]
        """
        try:
            return self.evaluate(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'__call__' method not implemented for this actionAngle module"
            )
    
    def evaluate(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle

        Returns
        -------
        x_vx : list
            A list containing the phase-space coordinates [x,vx]

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        return self.xvFreqs(j, angle, **kwargs)

    def xvFreqs(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency

        Parameters
        ----------
        j : float
            Action.
        angle : numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            Tuple containing:
                - x (numpy.ndarray): x-coordinate.
                - vx (numpy.ndarray): Velocity in x-direction.
                - Omega (float): Frequency.

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        amp = torch.sqrt(2.0 * j / self._omega)
        x = amp * torch.sin(angle)
        vx = amp * self._omega * torch.cos(angle)
        return torch.cat((x[None], vx[None]))

    def Freqs(self, j, **kwargs):
        """
        Return the frequency corresponding to a torus

        Parameters
        ----------
        j : scalar
            action

        Returns
        -------
        Omega : float
            frequency

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        return self._omega