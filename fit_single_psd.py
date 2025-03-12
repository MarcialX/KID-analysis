# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Homodyne system
# fit_single_psd.py
# Set of functions to fit a single PSD.
#
# Marcial Becerril, @ 22 July 2024
# Latest Revision: 10 Mar 2025, 16:31 UTC
#
# TODO list:
# + Improve the fit. log-log fit.
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


import sys

import numpy as np

from scipy.optimize import curve_fit

from matplotlib.pyplot import *

sys.path.append('./')
from physics.funcs import *
from misc.funcs import *


# F U N C T I O N S
# ------------------------------------
def guess_psd_params(freq, psd, gr_lims=[20, 50], amp_lims=[7.0e4, 8.5e4]):
    """
    Guess parameters to fit PSD.
    Parameters
    -----------
    freq_psd:       [array] frequency array [Hz].
    psd:            [array] power spectral density [Hz²/Hz].
    gr_lims:        [list] generation-recombination noise range calculation [Hz].
    amp_lims:       [list] amplifier noise range calculation [Hz].
    -----------
    """

    # G E N E R A T I O N - R E C O M B I N A T I O N   N O I S E
    # -------------------------------------
    idx_from = np.where(freq > gr_lims[0])[0][0]

    if len(np.where(freq < gr_lims[1])[0]) > 0:
        idx_to = np.where(freq < gr_lims[1])[0][-1]
    else:
        idx_to = -1

    assert len(psd[idx_from:idx_to]) > 0, f'Selected PSD array length is zero!\nCheck the defined gr limits.'

    # gr guessed
    gr_guess = np.median(psd[idx_from:idx_to])

    # define bounds
    gr_min = np.min(psd[idx_from:idx_to])
    gr_max = np.max(psd[idx_from:idx_to])

    # Q U A S I P A R T I C L E   L I F E T I M E
    # -------------------------------------
    # quasipartile lifetime
    tauqp_guess = 1./(np.max(freq)-np.min(freq))

    # define bounds
    tauqp_min = 1./np.max(freq)
    tauqp_max = 1./np.min(freq)

    # T L S   N O I S E 
    # -------------------------------------
    idx_one = np.where(freq > 1)[0][0]
    tlsa_guess = psd[idx_one]
    tlsb_guess = -0.5

    tlsa_min = np.min(psd)
    tlsa_max = np.inf

    tlsb_min = -1.5 #-0.7
    tlsb_max = -0.001
    
    # A M P   N O I S E
    # -------------------------------------
    amp_idx = np.where((psd >= amp_lims[0]) & (psd <= amp_lims[1]))[0]

    if len(amp_idx) > 0:
        amp_guess = np.median(psd[amp_idx])
        amp_min = np.min(psd[amp_idx])
        amp_max = np.max(psd[amp_idx])
    
    else:
        amp_guess = psd[-1]
        amp_min = -np.inf
        amp_max = np.inf

    # collect the guess
    guess = np.array([gr_guess, tauqp_guess, tlsa_guess, tlsb_guess, amp_guess])

    # collect the bounds
    bounds = np.array([ [gr_min, tauqp_min, tlsa_min, tlsb_min, amp_min],
                        [gr_max, tauqp_max, tlsa_max, tlsb_max, amp_max]])
    
    return guess, bounds


def fit_res_psd(freqs, psd, Q, f0, model='log', inter=True, **kwargs):
    """
    Fit a resonator power spectral density noise.
    Parameters
    ----------
    freqs:      [array] post-binned frequency array [Hz].
    psd:        [array] psd array [Hz²/Hz].
    Q:          [float] total quality factor.
    f0:         [float] resonance frequency [Hz].
    model:      [str] linear or log model.
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # generation-recombination noise range
    gr_lims = kwargs.pop('gr_lims', [20, 50])
    # amplifier noise range
    amp_lims = kwargs.pop('amp_lims', [7.0e4, 8.5e4])
    # save figure path
    save_path = kwargs.pop('save_path', './')
    # ----------------------------------------------
    
    assert model in ['lin', 'log'], "Not valid model.\nUse 'lin' linear or 'log' for logaritmic model."

    # guess psd parameters
    guess, bounds = guess_psd_params(freqs, psd, gr_lims=gr_lims, amp_lims=amp_lims)

    if model == 'lin':

        # frequencies weights
        sigma = (1 / abs(np.gradient(freqs)))

        try:
            # fit psd
            popt, pcov = curve_fit(lambda freqs, gr_level, tau_qp, tls_a, 
                                    tls_b, amp_noise : kid_spectra_noise_model(freqs, gr_level, 
                                                                                tau_qp, tls_a, tls_b, 
                                                                                amp_noise, Q, f0), 
                                                                                freqs, psd, bounds=bounds, 
                                                                                p0=guess, sigma=sigma)

            fitted_psd = kid_spectra_noise_model(freqs, *popt, Q, f0)
        
        except:
            popt, fitted_psd = None, None

    else:

        try:
            # fit psd
            popt, pcov = curve_fit(lambda lo_freqs, gr_level, tau_qp, tls_a, 
                                    tls_b, amp_noise : kid_spectra_noise_log_model(lo_freqs, gr_level, 
                                                                                tau_qp, tls_a, tls_b, 
                                                                                amp_noise, Q, f0), 
                                                                                np.log10(freqs), np.log10(psd), 
                                                                                bounds=bounds, p0=guess)

            fitted_psd = 10**(kid_spectra_noise_log_model(np.log10(freqs), *popt, Q, f0))

            printc('Log model used to fit PSD.', 'ok')

        except:
            popt, fitted_psd = None, None

    # interactive mode
    if inter:
        interFitPSD = fitPSDApp(freqs, psd, Q, f0, popt, fitted_psd, save_path=save_path)

        return (interFitPSD.gr_noise, interFitPSD.tau, interFitPSD.tls_a, interFitPSD.tls_b,
                interFitPSD.amp_noise), [interFitPSD._freq_psd, interFitPSD.fit_psd]
    
    else:
        return popt, [freqs, fitted_psd]


class fitPSDApp(object):
    """
    PSD fitting object.
    Handle the PSD fitting
    Parameters
    ----------
    freq_psd:       [array] frequency array [Hz].
    psd:            [array] power spectral density [Hz²/Hz].
    f0:             [float] resonance frequency [Hz].
    Q:              [float] total quality factor.
    ----------
    """
    def __init__(self, frequency, psd, Q, f0, popt, fitted_psd, model='log', **kwargs):

        # Key arguments
        # ----------------------------------------------
        # Lower limit amplifier noise
        #self.freq_amp = kwargs.pop('freq_amp', 7e4)
        # Project name
        self.save_path = kwargs.pop('save_path', './')
        # ----------------------------------------------

        # resonator model
        self.model = model

        # resonator parameters
        self.f0 = f0
        self.Q = Q

        # data
        self.frequency = frequency
        self.psd = psd

        # get initial fitted params
        self.fit_psd = fitted_psd

        # get psd parameters
        self.gr_noise = popt[0]
        self.tau = popt[1]
        self.tls_a = popt[2]
        self.tls_b = popt[3]
        self.amp_noise = popt[4]

        # interactive mode
        self.interactive_mode(frequency, psd, fitted_psd)


    # I N T E R A C T I V E   F U N C T I O N S 
    # ---------------------------
    def interactive_mode(self, frequency, psd, fitted_psd):
        """
        Interactive mode to clean psd data.
        Parameters
        ----------
        frequency:      [array] frequency array [Hz].
        psd:            [array] power spectral density [Hz²/Hz].
        fitted_psd:     [array] fitted PSD.
        ----------
        """

        # create figures
        self._fig = figure(figsize=(16, 10))
        subplots_adjust(left=0.075, right=0.975, top=0.95, bottom=0.075, wspace=0.12)

        self._ax = self._fig.add_subplot(111)

        self._freq_psd = frequency
        self._psd = psd
        self._fit_psd = fitted_psd

        self._cnt = 0
        self._idx = 0
        self.x_range = np.zeros(2, dtype=int)

        self.update_plot(self._freq_psd, self._psd, self._fit_psd)

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick_ipsd)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed_ipsd)

        show()


    def update_plot(self, frequency, psd, psd_fit):
        """
        Update interactive plot.
        Parameters
        ----------
        frequency:      [array] frequency array [Hz].
        psd:            [array] power spectral density [Hz²/Hz].
        psd_fit:        [array] fitted PSD.
        ----------
        """

        # clear subplot
        self._ax.clear()

        # plot raw PSD
        self._ax.loglog(frequency, psd, 'r')

        self._ax.set_xlabel(r'Frequency [Hz]', fontsize=18, weight='bold')
        self._ax.set_ylabel(r'PSD [Hz$^2$/Hz]', fontsize=18, weight='bold')

        self._ax.grid(True, which="both", ls="-")

        try:
            
            tau_t = self._tau
            gr_noise_t = self._gr_noise
            tls_a_t = self._tls_a
            tls_b_t = self._tls_b
            amp_noise_t = self._amp_noise

        except:

            tau_t = self.tau
            gr_noise_t = self.gr_noise
            tls_a_t = self.tls_a
            tls_b_t = self.tls_b
            amp_noise_t = self.amp_noise

        # generation-recombination noise
        gr = gr_noise(frequency, gr_noise_t, tau_t, self.Q, self.f0)
        
        # tls noise
        tls = tls_noise(frequency, tls_a_t, tls_b_t, self.Q, self.f0)

        self._ax.loglog(frequency, gr, 'm-', lw=2.5, label='gr noise')
        self._ax.loglog(frequency, tls, 'b-', lw=2.5, label='1/f')
        self._ax.loglog(frequency, amp_noise_t*np.ones_like(frequency), 'g-', label='amp', lw=2)

        # show 1/f knee
        knee = frequency[np.argmin( np.abs( gr_noise_t - tls ) )]

        self._ax.axvline(knee, color='c', linestyle='dashed', lw=2)

        self._ax.axhline(gr_noise_t, color='k', linestyle='dashed', lw=2)
        self._ax.axvline(1/(2*np.pi)/tau_t, color='orange', linestyle='dashed', lw=2)

        self._ax.text(0.05, 0.1, f'tau$_{{qp}}$: {tau_t*1e6:.1f} us\ngr noise: {gr_noise_t:.3f} Hz²/Hz\n'+ \
            f'amp noise: {amp_noise_t:.3f} Hz²/Hz\ntls$_a$: {tls_a_t:.3f} Hz²/Hz\ntls$_b$: {tls_b_t:.3f}\n'+ \
            f'1/f knee: {knee:.1f} Hz\nf0: {1e-6*self.f0:.1f} MHz\nQr: {self.Q:,.0f}',
            {'fontsize': 15}, bbox=dict(facecolor='orange', alpha=0.4), transform=self._ax.transAxes)

        # plot fitted data
        self._ax.semilogx(frequency, psd_fit, 'k', lw=2.5)


    def _key_pressed_ipsd(self, event):
        """
        Keyboard event to save/discard line fitting changes.
        Parameters
        ----------
        event:      [event] key pressed event.
        ----------
        """

        sys.stdout.flush()

        if event.key in ['x', 'q', 'd', 'w']:

            # save changes and close interactive mode.
            if event.key == 'x':
                
                self._fig.canvas.mpl_disconnect(self._onclick_ipsd)
                self._fig.canvas.mpl_disconnect(self._key_pressed_ipsd)

                self._fig.savefig(self.save_path)
                close(self._fig)
                
                try:
                    # save params
                    self.fit_psd = self._fit_psd

                    self.gr_noise = self._gr_noise
                    self.tau = self._tau
                    self.amp_noise = self._amp_noise
                    self.tls_a = self._tls_a
                    self.tls_b = self._tls_b

                    printc('Changes saved', 'ok')

                except:

                    printc('No changes', 'warn')

            # discard changes and close interactive mode
            elif event.key == 'q':
                
                self._fig.canvas.mpl_disconnect(self._onclick_ipsd)
                self._fig.canvas.mpl_disconnect(self._key_pressed_ipsd)

                self._fig.savefig(self.save_path)
                close(self._fig)
                
                printc('No changes to the fitting', 'info')

            # apply fit
            elif event.key == 'w':
                
                x_sort = np.sort(self.x_range)
                _freq_psd = np.concatenate((self._freq_psd[:x_sort[0]],self._freq_psd[x_sort[1]:]))

                # get amplifier noise
                #self.amp_noise = np.nanmedian(self._psd[np.where(_freq_psd>self.freq_amp)[0]])

                #if not np.isnan(self.amp_noise):
                    
                # R E P E A T   F I T
                # --------------------------
                # redefine freq and psd
                self._freq_psd = _freq_psd
                self._psd = np.concatenate((self._psd[:x_sort[0]],self._psd[x_sort[1]:]))

                # fit new curve
                popt, fitted_psd = fit_res_psd(self._freq_psd, self._psd, self.Q, self.f0, model=self.model, inter=False)

                # S A V E   P A R A M S
                # --------------------------
                self._fit_psd = fitted_psd[1]

                self._gr_noise = popt[0]
                self._tau = popt[1]
                self._tls_a = popt[2]
                self._tls_b = popt[3]
                self._amp_noise = popt[4]
                
                # U P D A T E   P L O T
                # --------------------------
                self.update_plot(self._freq_psd, self._psd, fitted_psd[1])
                self._fig.canvas.draw_idle()

                printc('Range removed', 'info')

                #else:
                #printc('Delete all data > ' + str(self.freq_amp) + 'Hz is not allowed', 'info')

            # discard changes and load initial values
            elif event.key == 'd':
                
                # reset values
                self._cnt = 0
                self._idx = 0
                self.x_range = np.zeros(2, dtype=int)

                # U P D A T E   P L O T
                # --------------------------
                self._freq_psd = self.frequency
                self._psd = self.psd

                self._gr_noise = self.gr_noise
                self._tau = self.tau
                self._amp_noise = self.amp_noise
                self._tls_a = self.tls_a
                self._tls_b = self.tls_b

                #cla()
                self.update_plot(self.frequency, self.psd, self.fit_psd)
                self._fig.canvas.draw_idle()
                
                printc('Reset interactive fit', 'info')


    def _onclick_ipsd(self, event):
        """
        On-click event to select lines
        Parameters
        ----------
        event:      [event] mouse click event.
        ----------
        """

        if event.inaxes == self._ax:

            # left-click select the data regions to discard.
            if event.button == 3:
               
                ix, iy = event.xdata, event.ydata
                
                # handle data out of range
                
                if ix>self._freq_psd[-1]:
                    xarg = len(self._freq_psd)
                
                else:
                    xarg = np.where(self._freq_psd>ix)[0][0]
                
                self._ax.axvline(ix)

                self._cnt += 1
                self._idx = self._cnt%2
                self.x_range[self._idx] = xarg

                self._fig.canvas.draw_idle()