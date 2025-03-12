# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID analysis. Useful physical functions 
# physical_funcs.py
# Some useful physical functions
#
# Marcial Becerril, @ 09 February 2025
# Latest Revision: 09 Feb 2025, 22:45 UTC
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import numpy as np
from scipy import integrate
from matplotlib.pyplot import *

from .physical_constants import *


# F U N C T I O N S
# -----------------------------------

def cr_time_response(x, a, t, c):
    """
    Cosmic rays time response.
    Parameters
    ----------
    x:  [array] timestream signal.
    a:  [float] amplitude.
    t:  [float] time constant [s].
    c:  [float] offset.
    ---------- 
    """
    return a*np.exp(-x/t) + c


def nep_gr(Nqp, tqp, Tc, eta=0.6):
    """
    Generation-Recombination noise.
    Parameters
    ----------
    Nqp:    [array] number of quasiparticles.
    tqp:    [float] quasiparticle lifetime [s].
    Tc:     [float] critical temperature [K].
    eta:    [float] conversion efficiency photon energy - quasiparticles.
    ----------
    """

    Delta = get_Delta(Tc)
    return (2*Delta/eta)*np.sqrt(Nqp/tqp)


def nep_rec(P, Tc, n0=0.4):
    """
    Optical Recombination noise.
    Parameters
    ----------
    P:      [array/float] optical power [W].
    Tc:     [float] critical temperature [K].
    n0:     [float] n0 constant, ~0.4 by default.
    ----------
    """

    Delta = get_Delta(Tc)
    return 2*np.sqrt(Delta*P/n0)


def nep_shot(P, f0, eta=0.6):
    """
    NEP shot noise.
    Parameters
    ----------
    P:      [array/float] optical power [W].
    f0:     [float] central frequency [Hz].
    eta:    [float] optical efficiency.
    ----------
    """

    return np.sqrt(2*h*f0*P/eta)


def nep_wave(P, dnu, modes=1):
    """
    NEP wave noise.
    Parameters
    ----------
    P:      [array/float] optical power [W].
    dnu:    [float] bandwidth [Hz].
    modes:  [int] number of modes.
    ----------
    """

    return np.sqrt(2*P**2/dnu/modes)


def nep_photon(P, dnu, f0, modes=1, eta=0.6):
    """
    NEP photon noise.
    Parameters
    ----------
    P:      [array/float] optical power [W].
    dnu:    [float] bandwidth [Hz].
    f0:     [float] central frequency [Hz].
    modes:  [int] number of modes.
    ----------
    """

    return np.sqrt( nep_shot(P, f0, eta=eta)**2 + nep_wave(P, dnu, modes=modes)**2 )


def tau_r(T, Tc, t0):
    """
    Get tau_r time as described in Pan et al. 2023.
    Parameters
    ----------
    T:      [float] base temperature [K].
    Tc:     [float] critical temperature [K].
    t0:     [float] material-dependent characteristic electron-phonon
            interaction time and can be modified by impurity 
            scattering.
    ----------
    """

    Delta = get_Delta(Tc)
    return (t0/np.sqrt(np.pi))*(Kb*Tc/(2*Delta))**(5/2) * np.sqrt(Tc/T) * np.exp(Delta/Kb/T)


def kid_spectra_noise_model(freqs, gr_level, tau_qp, tls_a, tls_b, amp_noise, Qr=20e3, f0=1e9):
    """
    KID spectra noise model.
    Parameters
    ----------
    freqs:          [array/float] frequency array [Hz]
    gr_level:       [float] generation-Recombination noise level [Hz²/Hz].
    tau_qp:         [float] quasiparticle lifetime [s]. 
    tls_a, tls_b :  [float] parameters that make up the 1/f noise component.
    Qr:             [float] total quality factor.
    f0:             [float] resonance frequency [Hz].
    amp_noise:      [float] amplifier noise [Hz²/Hz].
    ----------
    """

    # get generation-recombination noise
    gr = gr_noise(freqs, gr_level, tau_qp, Qr, f0)

    # get tls noise
    tls = tls_noise(freqs, tls_a, tls_b, Qr, f0)

    # add all the noise sources
    return amp_noise + gr + tls


def kid_spectra_noise_log_model(log_freqs, gr_level, tau_qp, tls_a, tls_b, amp_noise, Qr=20e3, f0=1e9):
    """
    KID spectra noise log model.
    Parameters
    ----------
    log_freqs:      [array/float] log frequency array [Hz]
    gr_level:       [float] generation-Recombination noise level [Hz²/Hz].
    tau_qp:         [float] quasiparticle lifetime [s]. 
    tls_a, tls_b:   [float] parameters that make up the 1/f noise component.
    Qr:             [float] total quality factor.
    f0:             [float] resonance frequency [Hz].
    amp_noise:      [float] amplifier noise [Hz²/Hz].
    ----------
    """

    freqs = 10**log_freqs

    # get generation-recombination noise
    gr = gr_noise(freqs, gr_level, tau_qp, Qr, f0)

    # get tls noise
    tls = tls_noise(freqs, tls_a, tls_b, Qr, f0)

    # add all the noise sources
    return np.log10( amp_noise + gr + tls )


def gr_noise(freqs, gr_level, tau_qp, Qr, f0):
    """
    Generation-Recombination noise.
    Parameters:
    -----------
    freqs:      [array/float] frequency array [Hz]
    gr_noise:   [float] generation-Recombination noise level [Hz²/Hz].
    tau_qp:     [float] quasiparticle lifetime [s].
    Qr:         [float] total quality factor.
    f0:         [float] resonance frequency [Hz].
    -----------
    """

    decay = response_decay(freqs, tau_qp, Qr, f0)
    
    return gr_level * decay


def tls_noise(freqs, tls_a, tls_b, Qr, f0):
    """
    Generation-Recombination noise.
    Parameters:
    -----------
    freqs:          [array/float] frequency array [Hz]
    tls_a, tls_b:   [float] parameters that make up the 1/f noise component.
    Qr :            [float] total quality factor.
    f0 :            [float] resonance frequency [Hz].
    -----------
    """

    # if ring-down won't be considered
    if f0 == 0 and Qr == 0:
        f0, Qr = 1, 0
        print(f'Ring-down time ignored.')

    return (tls_a*freqs**(tls_b)) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)


def response_decay(freqs, tau_qp, Qr, f0):
    """ Compute the response decay. """
    
    # if ring-down won't be considered
    if f0 == 0 and Qr == 0:
        f0, Qr = 1, 0
        print(f'Ring-down time ignored.')

    return 1 / (1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)


def f0_vs_pwr_model(P, a, b):
    """
    Responsivity model as a power-law with an 'b' index.
    Parameters
    ----------
    P: [array/float]power in Watts.
    a: [float] proportionality constant.
    b: [float] power-law index
    ----------
    """

    return a*P**b


def bb2pwr(T, nu):
    """
    Get the power from the blackbody assuming a throughput (A*Omega) equals to 1.
    Parameters
    ----------
    T:  [array/float] blackbody temperature [K].
    nu: [float] bandwidth [Hz].
    ----------
    """

    return Kb * np.array(T) * nu


def planck(nu, T):
    """
    From Tom Brien.
    Defines Planck function in frequency space.
    Parameters
    ----------
    nu: [array/float] frequency [Hz].
    T:  [float] blackbody temperature [K].
    ----------
    """

    return 2*h*nu**3/c**2 * 1 / (np.exp((h*nu)/(Kb*T)) - 1)


def optical_NEP(f, Sa, tqp, S, Qr, f0):
    """
    Get optical NEP.
    Parameters
    ----------
    f:      [array/float] frequency array [Hz].
    Sa:     [float] power Spectrum Density [Hz²/Hz].
    tqp:    [float] quasiparticle lifetime [s].
    S:      [float] responsivity [Hz/W].
    Qr:     [float] total quality factor.
    f0:     [float] resonance frequency [Hz].
    ----------
    """

    return np.sqrt(Sa) * ( (np.abs(S))**(-1)) * \
        np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)


def dark_NEP(f, Sa, tqp, S, Qr, f0, Delta, eta=0.6):
    """
    Get the dark NEP.
    Parameters
    ----------
    f:          [array/float] frequency array [Hz].
    Sa:         [float] power Spectrum Density [Hz²/Hz].
    tqp:        [float] quasiparticle lifetime [s].
    S:          [float] responsivity [Hz/W].
    Qr:         [float] total quality factor.
    f0:         [float] resonance frequency [Hz].
    Delta:      [float] energy gap [J].
    eta(opt):   [float] optical efficiency.
    ----------
    """

    return np.sqrt(Sa) * (( (eta*tqp/Delta)*(np.abs(S)) )**(-1)) * \
          np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)


def get_Delta(Tc):
    """
    Get energy gap, for a T << Tc.
    Parameters
    ----------
    Tc: [float] critical temperature [K].
    ----------
    """

    return 3.528*Kb*Tc/2


def get_nqp(N0, T, Delta):
    """
    Get the quasiparticle density.
    Parameters
    ----------
    N0:     [float] single spin density of states at the Fermi level.
    T:      [float] base temperature [K].    
    Delta:  [float] energy gap [J].
    ----------
    """

    return 2 * N0 * np.sqrt( 2*np.pi*Kb*T*Delta ) * np.exp(-(Delta/(Kb*T)))


def n_occ(freq, T):
    """
    Photon occupation number as defined in
    https://github.com/chill90/BoloCalc
    Parameters
    ----------
    freq:   [array/float] frequency [Hz].
    T:      [float] temperature [K].
    ----------
    """

    return 1/(np.exp((h*freq)/(Kb*T))-1)


def dPdT(freq, tx):
    """
    Incident power fluctuations due to fluctuations
    in CMB temperature.
    Ref: Hill et al 2019 (SO Collaboration)
    Parameters
    ----------
    freq:   [array/float] frequency [Hz]
    tx:     [float] transmission
    ----------
    """

    a = (1/Kb) * ((h*freq/Tcmb)**2) * (n_occ(freq, Tcmb)**2) * np.exp((h*freq)/(Kb*Tcmb)) * tx
    return integrate.trapezoid(a, freq)


def load_fts(fts_path, freq_range=[0, 350], **kwargs):
    """
    Read and load fts measurements.
    Parameters
    ----------
    fts_path:           [str] fts path file [from Tom Brien format].
    freq_range:         [float] selection frequency range [GHz].
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # frequency column
    freq_col = kwargs.pop('freq_col', 0)
    # tx column
    tx_col = kwargs.pop('tx_col', 1)
    # ----------------------------------------------

    # load transmission.
    fts = np.load(fts_path, allow_pickle=True)

    freq = fts[:, freq_col][1:]         # frequency
    tx = fts[:, tx_col][1:]             # transmission

    # filter transmission, just select up to an upper limit.
    sel_index = np.where( (freq >= freq_range[0]) & (freq <= freq_range[1]) )[0]
    
    freq_sel = freq[sel_index]
    tx_sel = tx[sel_index]

    print(f'Selected frequency range: {freq_sel[0]:.1f}-{freq_sel[-1]:.1f} Hz')

    return freq_sel, tx_sel


def AOmega(f0, modes=1):
    """
    Get throughput, AOmega.
    Parameters
    ----------
    f0:     [array/float] central frequency [Hz].
    modes:  [int] number of modes.
            1 : for single-polarization detectors.
            2 : for dual-polarization detectors.
    ----------
    """

    # assuming beam filled.
    return modes*((c/f0)**2)/2


def get_power_from_FTS(freq, tx, T, modes=1):
    """
    Get the power from the FTS.
    Parameters
    ----------
    freq:   [array] frequency array in GHz.
    tx:     [array] spectral transmission.    
    T:      [float] blackbody temperature [K].
    modes:  [int] number of modes.
    ----------
    """

    spectrum = tx * planck(1e9*freq, T)     # modulated by the bb temperature.
    
    AO = AOmega(1e9*freq, modes=modes)     # get A0

    # get total power
    return integrate.trapezoid(AO*spectrum, 1e9*freq)


def gaussian(x, A, mu, sigma, offset):
    """
    Gaussian function
    Parameters
    ----------
    x:      [array/float] x-array.
    A:      [float] amplitude.
    mu:     [float] mean.
    sigma:  [float] dispersion.
    offset: [float] offset.
    ----------
    """
    return offset+A*np.exp(-((x-mu)**2)/(2*sigma**2))


def twoD_Gaussian(pos, amplitude, xo, yo, sigma, offset):
    """
    2D Gaussian function.
    Parameters
    ---------
        Input:
            pos:     	[list] Points of the 2D map
            amplitude:  [float] Amplitude
            xo, yo:     [float] Gaussian profile position
            sigma:      [float] Dispersion profile
            offset:     [float] Offset profile
        Ouput:
            g:			[list] Gaussian profile unwraped in one dimension
    ---------
    """
	
    x = pos[0]
    y = pos[1]
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude*np.exp(-(((x - xo)**2/2./sigma**2) + ((y - yo)**2/2./sigma**2)))

    return g.ravel()


def twoD_ElGaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Elliptical Gaussian function.
    Parameters
    ---------
        Input:
            pos:             	[list] Points of the 2D map
            amplitude:          [float] Amplitude
            xo, yo:             [float] Gaussian profile position
            sigma_x, sigma_y:   [float] X-Y Dispersion profile
            theta:              [float] Major axis inclination
            offset:             [float] Offset profile
        Ouput:
            g:			[list] Gaussian profile unwraped in one dimension
    ---------
    """
    
    x = pos[0]
    y = pos[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset+amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()
