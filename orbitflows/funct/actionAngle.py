import numpy as np
import matplotlib.pyplot as plt
import torch


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