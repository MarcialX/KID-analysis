# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. Fit resonators functions
# fit_resonators.py
# Set of functions to fit resonators.
#
# Marcial Becerril, @ 23 February 2025
# Latest Revision: 23 Feb 2025, 01:25 UTC
# 
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import os
import sys
import lmfit

import numpy as np

from scipy.signal import savgol_filter

from sweep_functions import *
from misc.funcs import *



# R E S O N A T O R   M O D E L   F U N C T I O N S
# ---------------------------------------------------------
def cable_delay(frequency, ar, ai, **kwargs):
    """
    Cable delay.
    Parameters
    ----------
    frequency:      [array] frequency [Hz].
    ar, ai:         [array] complex amplitude.
    tau[opt]:       [float] delay constant [s].
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    # Cable delay
    return (ar+1j*ai)*np.exp(-1j*2*np.pi*frequency*tau)


def resonator_model(frequency, fr, ar, ai, Qr, Qc, phi, non, **kwargs):
    """
    Resonators nonlinear model.
    From Swenson et al. 2013: https://arxiv.org/abs/1305.4281
    Parameters
    ----------
    frequency:      [array] frequency [Hz].
    fr:             [float] resonance frequency [Hz].
    ar, ai:         [array] complex amplitude.
    Qr:             [float] total quality factor.
    Qc:             [float] coupling quality factor.
    phi:            [float] rotation circle [rad].
    non:            [float] nonlinear parameter.
    tau[opt]:       [float] cable delay [s].
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    A = (ar + 1j*ai)*np.exp(-1j*2*np.pi*frequency*tau)

    # Fractional frequency shift of non-linear resonator
    y0s = Qr*(frequency - fr)/fr
    y = np.zeros_like(y0s)

    for i, y0 in enumerate(y0s):
        coeffs = [4.0, -4.0*y0, 1.0, -(y0 + non)]
        y_roots = np.roots(coeffs)
        # Extract real roots only [From Pete Barry's Code.]
        # If all the roots are real, take the maximum
        y_single = y_roots[np.where(abs(y_roots.imag) < 1e-5)].real.max()
        y[i] = y_single

    B = 1 - (Qr/(Qc*np.exp(1j*phi))) / (1 + 2j*y )

    return A*B


def phase_angle(frequency, theta_0, fr, Qr):
    """
    Phase angle.
    Parameters
    ----------
    frequency:      [array] frequency [Hz].
    theta_0:        [array] circle rotation [rad].
    fr:             [float] resonance frequency [Hz].
    Qr:             [float] total quality factor.
    ----------
    """

    # get phase angle
    theta = -theta_0 + 2*np.arctan( 2*Qr*(1 - (frequency/fr) ) )
    
    return theta


def guess_resonator_params(frequency, data, tau=50e-9):
    """
    Guess initial parameters to fit resonator.
    Based on Gao PhD thesis.
    Parameters
    ----------
    frequency:      [array] frequency [Hz].
    data:           [array] sweep complex data.
    tau[opt]:       [float] cable delay [s].
    ----------
    """

    # get I,Q
    I = data.real
    Q = data.imag

    # remove the cable delay
    # ------------------------------------------
    cable_delay_model = CableDelay(tau=tau)

    freq_cable = np.concatenate((frequency[:10], frequency[-10:]))
    I_cable = np.concatenate((I[:10], I[-10:]))
    Q_cable = np.concatenate((Q[:10], Q[-10:]))

    s21_cable = I_cable + 1j*Q_cable

    guess = cable_delay_model.guess(freq_cable, s21_cable)
    cable_res = cable_delay_model.fit(s21_cable, params=guess, frequency=freq_cable)

    fit_s21 = cable_delay_model.eval(params=cable_res.params, frequency=frequency)

    ar = cable_res.values['cd_ar']
    ai = cable_res.values['cd_ai']

    s21_no_cable = data/fit_s21

    I_no_cable = s21_no_cable.real
    Q_no_cable = s21_no_cable.imag

    # derotate
    # ------------------------------------------
    idx_f0 = np.argmin( np.abs(data) )

    f0n = frequency[idx_f0]
    I0 = I_no_cable[idx_f0]
    Q0 = Q_no_cable[idx_f0]

    sel = np.abs(frequency-f0n) < 10e4
    xc, yc, r = fit_circ(I_no_cable[sel], Q_no_cable[sel])

    theta = np.arctan2(Q0-yc, I0-xc)
    I_derot = (I_no_cable-xc)*np.cos(-theta)-(Q_no_cable-yc)*np.sin(-theta)
    Q_derot = (I_no_cable-xc)*np.sin(-theta)+(Q_no_cable-yc)*np.cos(-theta)

    # fit the phase angle
    # ------------------------------------------
    sel2 = np.abs(frequency - f0n) < 400e3
    phase_angle_model = PhaseAngle()

    phase = get_phase(I_derot[sel2] + 1j*Q_derot[sel2])

    guess = phase_angle_model.guess(frequency[sel2], phase)

    phase_res = phase_angle_model.fit(phase, params=guess, frequency=frequency[sel2])
    fit_phase = phase_angle_model.eval(params=phase_res.params, frequency=frequency[sel2])

    Qr = phase_res.values['pa_Qr']
    fr = phase_res.values['pa_fr']
    theta_0 = phase_res.values['pa_theta_0']

    # get Qc
    # ------------------------------------------
    mag_zc = np.sqrt(xc**2 + yc**2)
    arg_zc = np.arctan2(yc, xc)

    Qc = Qr*(mag_zc + r)/(2*r)
    phi = theta_0 - arg_zc

    return ar, ai, Qr, fr, Qc, phi


# R E S O N A T O R   M O D E L I N G   C L A S S E S
# ---------------------------------------------------------
class ResGaoModel(lmfit.model.Model):
    __doc__ = "resonator Gao model" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='rgm_', *args, **kwargs):
        """
        Gao's resonator model class.
        Parameters
        ----------
        prefix:     [string] model prefix. 'rgm_' by default.
        ----------
        """
        super().__init__(resonator_model, *args, **kwargs)

        # Key arguments
        # ----------------------------------------------
        # Cable delay
        self.tau = kwargs.pop('tau', 50e-9)
        # ----------------------------------------------

        self.prefix = prefix


    def guess(self, frequency, data, **kwargs):
        """
        Guessing resonator parameters.
        Parameters
        ----------
        frequency:      [array] frequency [Hz]
        data:           [array] S21 sweep data.
        ----------
        """

        ar, ai, Qr, fr, Qc, phi = guess_resonator_params(frequency, data, tau=self.tau)

        # defining the boundaries
        self.set_param_hint('%sar' % self.prefix, value=ar)
        self.set_param_hint('%sai' % self.prefix, value=ai)

        self.set_param_hint('%sQr' % self.prefix, value=Qr) #, min=100)
        #self.set_param_hint('%stau' % self.prefix, value=50e-9, min=40e-9, max=60e-9)
        self.set_param_hint('%sfr' % self.prefix, value=fr, min=frequency[0], max=frequency[-1])
        #self.set_param_hint('%stheta_0' % self.prefix, value=theta_0, min=-20*np.pi, max=20*np.pi)
        # this is delta number to asure Qc is always positive
        #self.set_param_hint('%sdelta' % self.prefix, value=50e3, min=0, vary=True)

        #self.set_param_hint('%sQc' % self.prefix, expr='%sdelta' % self.prefix+'+'+'%sQr' % self.prefix)
        self.set_param_hint('%sQc' % self.prefix, value=Qc)
        self.set_param_hint('%sphi' % self.prefix, value=phi, min=-20*np.pi, max=20*np.pi)
        self.set_param_hint('%snon' % self.prefix, value=0.1, min=0.0, max=3.0)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class CableDelay(lmfit.model.Model):
    __doc__ = "cable delay" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='cd_', *args, **kwargs):
        """
        Cable delay model class.
        Parameters
        ----------
        prefix:     [string]  model prefix. 'cd_' by default
        ----------
        """
        super().__init__(cable_delay, *args, **kwargs)
        self.prefix = prefix


    def guess(self, frequency, data, **kwargs):
        """
        Guessing resonator parameters.
        Parameters
        ----------
        frequency:      [array] frequency [Hz].
        data:           [array] S21 sweep data.
        ----------
        """

        # amplitude
        ar_guess = np.mean(data.real)
        ai_guess = np.mean(data.imag)

        # defining boundaries
        self.set_param_hint('%sar' % self.prefix, value=ar_guess)#, min=np.min(data.real), max=np.max(data.real))
        self.set_param_hint('%sai' % self.prefix, value=ai_guess)#, min=np.min(data.imag), max=np.max(data.imag))

        # load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class PhaseAngle(lmfit.model.Model):
    __doc__ = "phase angle" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='pa_', *args, **kwargs):
        """
        Phase Angle model class.
        Parameters
        ----------
        prefix:     [str] model prefix. 'pa_' by default
        ----------
        """
        super().__init__(phase_angle, *args, **kwargs)
        self.prefix = prefix


    def guess(self, frequency, data, *args, **kwargs):
        """
        Guessing phase parameters.
        Parameters
        ----------
        frequency:      [array] frequency [Hz].
        data:           [array] S21 sweep data.
        ----------
        """

        # total Q
        Qr_guess = 2.5e4
        
        smooth_points = 31
        if len(frequency) < 50:
            smooth_points = int(len(frequency)/10)
            if smooth_points <= 3:
                smooth_points = 5

        # resonance frequency
        sm_speed_sweep = np.abs(savgol_filter( np.gradient(data, frequency), smooth_points, 3))
        f0_idx = np.argmax(sm_speed_sweep)
        fr_guess = frequency[f0_idx]
        
        # theta delay
        theta_0_guess = 0

        # defining boundaries
        self.set_param_hint('%sQr' % self.prefix, value=Qr_guess, ) #min=100, )
        self.set_param_hint('%sfr' % self.prefix, value=fr_guess, min=frequency[0], max=frequency[-1])
        self.set_param_hint('%stheta_0' % self.prefix, value=theta_0_guess, min=-20*np.pi, max=20*np.pi)

        # load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
