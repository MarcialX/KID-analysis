# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. VNA measurements class
# vna_meas.py
# Class to handle vna measurements.
#
# Marcial Becerril, @ 09 February 2025
# Latest Revision: 09 Feb 2025, 20:00 UTC
#
# TODO list:
#   - Get resonator dip-depth
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


import numpy as np

from scipy.signal import savgol_filter
from scipy import interpolate

from misc.funcs import *
from physics.funcs import *
from sweep_functions import *



# S W E E P   C L A S S E S
# -----------------------------------
class sweep:
    """ s21 measurements class. """

    def __init__(self, frequency, I, Q, hdr=None, f0_method='max-speed',
                 f0_smooth_params=[31, 3]):
        """
        Parameters
        ----------
        frequency:              [array] frequency array.
        I/Q:                    [array] S21 data as I/Q data or single complex array.
        hdr[opt]:               [dict] header file as dictionary.
        f0_method[opt]:         [str] f0 method in case f0 it is not defined.
        f0_smooth_params[opt]:  [list] smooth parameters to get f0.
        ----------
        """

        # check frequency data format
        assert isinstance(frequency, list) or isinstance(frequency, np.ndarray), \
            "Frequency data format not valid."

        self.frequency = frequency

        # check s21 data format
        assert isinstance(I, list) or isinstance(I, np.ndarray), \
            "I data format not valid."
        
        assert isinstance(Q, list) or isinstance(Q, np.ndarray), \
            "Q data format not valid."

        self.s21 = I + 1j*Q             # get S21 as complex array  
        self.s21_nr = self.s21.copy()   # s21 backup    
        self.hdr = hdr                  # get header

        if len(self.frequency) > 0:

            # if f0 is not available, get it from the measurements
            if not 'F0' in self.hdr:
                
                printc(f"f0 is not defined, computing f0 from data.", 'warn')
                
                try:
                    self.get_f0(method=f0_method, smooth_params=f0_smooth_params)   # get f0

                except:
                    printc(f'No possible to get f0, set as the freq mid point', 'fail')

            f0 = self.hdr['F0']
            printc(f"f0 found: {f0} Hz", 'info')

            if not 'I0' in self.hdr or not 'Q0' in self.hdr:
                self.IQ_at_f0(f0)                                               # get I0/Q0 

            if (not 'DIDF' in self.hdr) or (not 'DQDF' in self.hdr):
                self.compute_dxdf(smooth_params=f0_smooth_params)               # get didf, dqdf

            # get frequency range and center
            range = self.frequency[-1] - self.frequency[0]
            centre = self.frequency[int(len(self.frequency)/2)]

            self.hdr['BANDWIDT'] = range
            self.hdr['CENTRE'] = centre

            # rotation flag
            self.rotated = False

            # get circle position and rotation angle
            try:
                self.get_circle_pos_and_rot()

            except Exception as e:
                printc(f"Circle couldn't be derotated.", "warn")

            # phase variables
            self.phase = get_phase(self.s21)
            self.phase_mdl = None
            self.lineal_phase_coeffs = None

            # initialise fit dictionary
            self.fit = {}


    @property
    def lin2dB(self):
        """ Convert S21 data to dB. """
        return 20*np.log10(np.abs(self.s21))


    def IQ_at_f0(self, f0):
        """
        Get I0/Q0 at f0. 
        Parameters
        ----------
        f0:     [float] resonance frequency
        ----------
        """

        freq = self.frequency                   # get frequency
        I, Q = self.s21.real, self.s21.imag     # get I/Q

        try:
            idx = np.where(freq >= f0)[0][0]
        
        except:
            if freq[-1] < f0:
                idx = -1
            elif freq[0] > f0:
                idx = 0

        self.hdr['IF0'], self.hdr['QF0'] = I[idx], Q[idx]


    def get_circle_pos_and_rot(self, f0_thresh=8e4):
        """
        Get IQ circle position and angle rotation.
        Parameters
        ----------
        f0_thresh[opt]:     [float] threshold around f0 to select IQ data [Hz].
                            By default it is 80 kHz.
        ----------
        """

        if self.rotated:
            printc('WARNING! circle has been rotated!', 'warn')

        f0 = self.hdr['F0']

        freq = self.frequency                   # get frequency
        I, Q = self.s21.real, self.s21.imag     # get I/Q

        assert f0 >= freq[0] and f0 <= freq[-1], "f0 out of frequency range.\nf0 has to be within the frequency range"

        I0 = self.hdr['IF0']                    # I0 at resonance frequency
        Q0 = self.hdr['QF0']                    # Q0 at resonance frequency

        sel = np.abs(freq - f0) < f0_thresh
        if len(sel) == 0:
            printc(f'No data selected within the threshold.\n'+
                    'Select another f0 or change the threshold', 'fail')
            return

        assert len(I[sel]) > 1, "Insufficient selected points, try changing the threshold"

        xc, yc, r = fit_circ(I[sel], Q[sel])    # get circle center and radius.

        theta = np.arctan2(Q0 - yc, I0 - xc)    # get rotation angle.

        self.xc, self.yc, self.theta = xc, yc, theta
        #return xc, yc, theta


    def derotate_circle(self):
        """ Get IQ circle rotation. """

        xc, yc, theta = self.xc.copy(), self.yc.copy(), self.theta.copy()

        # derotate data
        I, Q = self.s21.real, self.s21.imag 
        Is_derot, Qs_derot = derot_circle(I, Q, xc, yc, theta)

        self.s21 = Is_derot + 1j*Qs_derot

        # update phase
        self.phase = get_phase(self.s21)

        # set flag rotation
        self.rotated = True
        printc(f'Circle derotated!', 'ok')

        #return s21_derot, xc, yc, theta


    def get_f0(self, method='max-speed', smooth_params=None):
        """
        Get resonance frequency from the measured data.
        Parameters
        ----------
        method:     [str] resonance frequency method.
                    Maximum IQ circle speed by default.
        ----------
        """

        methods = ['max-speed', 'min-s21', 'max-s21']
        assert method in methods, f"{method} is not valid method"

        freq = self.frequency
        s21 = self.s21

        if not smooth_params is None:
            
            printc('Smoothing I/Q signals', 'info')
            
            assert isinstance(smooth_params, list), "Smooth parameters has to be a list."
            assert len(smooth_params) == 2, "Smooth parameters requires two elements: number of points and order."

            s21 = self.smooth_s21(s21, smooth_params=smooth_params)

        if method == methods[0]:        # <--- maximum IQ speed method
            speed = get_circle_speed(freq, s21)    # get IQ circle speed
            f0_idx = np.nanargmax(speed)                # found index of max speed
        
        elif method == methods[1]:       # <--- minimum S21 amplitude
            f0_idx = np.argmin(np.abs(s21))
        
        elif method == methods[2]:       # <--- maximum S21 amplitude
            f0_idx = np.argmax(np.abs(s21))

        else:
            printc('Method not defined', 'warn')
            return

        self.hdr['F0'] = freq[f0_idx]     


    # C H E C K! This function may be restructured.
    def smooth_s21(self, s21, smooth_params=[31, 3]):
        """ Smooth S21 data. """

        I, Q = s21.real, s21.imag               # get I,Q components.

        npts = smooth_params[0]
        norder = smooth_params[1]
        
        I_sm = savgol_filter(I, npts, norder) # 11, 3
        Q_sm = savgol_filter(Q, npts, norder) # 11, 3

        return I_sm + 1j*Q_sm


    def phase_model(self, smooth_params=[31, 3], 
                    f0_phase_thresh=25e4, linear_fit=False, 
                    f0_thresh_linear_fit=5e4):
        """
        Get resonance frequency from phase.
        Parameters
        ----------
        smooth_params:          [list] smooth parameters for phase curve:
                                0: number of points, 1: order.
        f0_phase_thresh:        [float] phase threshold around f0.
        linear_fit:             [bool] fit a line to the phase.
        f0_thresh_linear_fit:   [float] freq data selected for the phase linear fit.
        ----------
        """

        # fit parameters, only if linear fit was selected.
        fit_coeffs = None

        f0 = self.hdr['F0']         # resonance frequency

        # get frequency and phase
        freq = self.frequency
        phase = self.phase

        # smooth phase data
        npts = smooth_params[0]
        norder = smooth_params[1]
        phase_sm = savgol_filter(phase, npts, norder) # 11, 3

        # mask a selected region
        freq_mask = np.abs(freq - f0) < f0_phase_thresh
        # masked phase
        phase_masked = phase_sm[freq_mask]
        f0_masked = freq[freq_mask]

        # fit a linear function around f0.
        if linear_fit:

            sel_phase = np.abs(freq - f0) < f0_thresh_linear_fit
            fit_coeffs = np.polyfit(phase_sm[sel_phase], freq[sel_phase], deg=1)

            phase_lin_fnc = np.poly1d(fit_coeffs)           # get fit linear model
            phase_masked = phase_sm[sel_phase]
            f0_masked = phase_lin_fnc(phase_masked)         # masked phase

        # interpolate phase, get phase vs df model
        phase2df_mdl = interpolate.interp1d(phase_masked, f0_masked, fill_value='extrapolate')

        self.phase_mdl = phase2df_mdl
        self.lineal_phase_coeffs = fit_coeffs
        #return phase2df_mdl, fit_coeffs 


    def get_dxdf(self, f0, freq, x):
        """
        Get the gradient dx/df at f0, where x could be I or Q signal.
        Parameters
        ----------
        f0:         [float] resonance frequency [Hz].
        freq:       [array] frequency array [Hz].
        x:          [array] I or Q signal.
        ----------
        """

        # resonance frequency index
        f0_idx = np.argmin(np.abs(freq - (f0)))
        # get dx/df at f0
        dxdf = np.gradient(x, freq)[f0_idx]

        return dxdf


    def compute_dxdf(self, smooth_params=None):
        """
        Calculate didf and dqdf at f0.
        Parameters
        ----------
        smooth_params:      [list] smooth parameters for I/Q curves
        ----------
        """

        freq = self.frequency       # frequency array
        f0 = self.hdr['F0']         # get resonance frequency

        s21 = self.s21              # get sweep

        if not smooth_params is None:
            
            printc('Smoothing I/Q signals', 'info')
            
            assert isinstance(smooth_params, list), "Smooth parameters has to be a list."
            assert len(smooth_params) == 2, "Smooth parameters requires two elements: number of points and order."

            s21 = self.smooth_s21(s21, smooth_params=smooth_params)

        self.hdr['DIDF'] = self.get_dxdf(f0, freq, s21.real)
        self.hdr['DQDF'] = self.get_dxdf(f0, freq, s21.imag)

        printc(f'di/df: {self.hdr['DIDF']}\ndq/df: {self.hdr['DQDF']}', 'ok')


    def get_dip_depth(self, select_edges=[50, -50], smooth_params=[15, 3]):
        """
        Get dip-depth from measured data.
        Parameters
        ----------
        select_edges:       [list] edges border to extract the baseline.
        smooth_params:      [list] smooth parameters.
        ----------
        """
        
        # get frerquency and s21
        freq = self.frequency
        s21_dB = self.lin2dB

        # remove baseline
        freq_baseline = np.concatenate( (freq[:select_edges[0]], freq[select_edges[1]:] ) )
        s21_dB_baseline = np.concatenate( (s21_dB[:select_edges[0]], s21_dB[select_edges[1]:] ) )

        fit_coeffs = np.polyfit(freq_baseline, s21_dB_baseline, deg=1)
        baseline_linear_model = np.poly1d(fit_coeffs)
        baseline = baseline_linear_model(freq)

        # remove baseline
        s21_dB_clear = s21_dB - baseline

        # smooth just a bit the s21 params
        s21_dB_clear_sm = savgol_filter(s21_dB_clear, smooth_params[0], smooth_params[1])
        
        # get the dip-depth
        dip_depth = np.min(s21_dB_clear_sm)

        return dip_depth


    def show_meas_properties(self):
        """
        Show basic information about data taken.
        """

        printc('S U M M A R Y   S W E E P', 'title3')
        print('------------------------------')
        
        if self.hdr['SWEEPTYP'][:3].lower() in 'seg':
            sweep_mode = 'Segmented'
        elif self.hdr['SWEEPTYP'][:3].lower() in 'con':
            sweep_mode = 'Continuous'
        else:
            sweep_mode = self.hdr['SWEEPTYP']
        printc(f'{sweep_mode} sweep', 'title2')

        printc(f'Date: {self.hdr['DATE']}\tTime: {self.hdr['TIME']}', 'info')
        printc(f'DUT: {self.hdr['DUT']}', 'info')
        printc(f'Room Att: {self.hdr['ATT_RT']} [dB]', 'ok')
        print('------------------------------')
        printc(f'Blackbody: {self.hdr['BLACKBOD']} [K]', 'ok')
        printc(f'Base temperature: {1e3*self.hdr['SAMPLETE']:.1f} [mK]', 'ok')
        print('------------------------------')
        printc(f'F0: {self.hdr['F0']} [Hz]', 'ok')
        printc(f'I0: {self.hdr['IF0']:.5f}', 'ok')
        printc(f'Q0: {self.hdr['QF0']:.5f}', 'ok')
        printc(f'Bandwidth: {1e-6*self.hdr['BANDWIDT']:.2f} [MHz]', 'ok')
        printc(f'Range: {1e-6*self.frequency[0]:.2f} to {1e-6*self.frequency[-1]:.2f} [MHz]', 'ok')
        printc(f'Central frequency: {1e-6*self.hdr['CENTRE']:.2f} [MHz]', 'ok')
        
        freq_step = np.mean(np.diff(self.frequency))
        printc(f'mean freq step: {freq_step:.2f} [Hz]', 'ok')

        print('------------------------------')


    def plot_sweep_gradients(self):
        """
        Plot IQ + gradients at the resonance frequency.
        """

        fig, axs = subplots(1, 2)
        # plot I
        axs[0].plot(self.frequency, self.s21.real, 'b.-')
        # plot didf gradient at f0
        b_I = self.hdr['IF0'] - self.hdr['DIDF']*self.hdr['F0']
        line_i = self.hdr['DIDF']*self.frequency + b_I
        axs[0].plot(self.frequency, line_i, 'r')

        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('I [V]')
        axs[0].set_title('I')
        axs[0].set_ylim([np.min(self.s21.real), np.max(self.s21.real)])
        axs[0].grid()

        # plot Q
        axs[1].plot(self.frequency, self.s21.imag, 'b.-')
        # plot dqdf gradient at f0
        b_Q = self.hdr['QF0'] - self.hdr['DQDF']*self.hdr['F0']
        line_q = self.hdr['DQDF']*self.frequency + b_Q
        axs[1].plot(self.frequency, line_q, 'r')

        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Q [V]')
        axs[1].set_title('Q')
        axs[1].set_ylim([np.min(self.s21.imag), np.max(self.s21.imag)])
        axs[1].grid()


    def plot_sweep(self):
        """ Plot s21 sweep. """

        fig, axs = subplots(1, 1)
        
        # plot sweep
        axs.plot(self.frequency, self.lin2dB, 'b.-')

        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('S21 [dB]')
        axs.set_title('S21 sweep')
        axs.grid()


    def plot_circle(self):
        """ Plot IQ sircle. """

        fig, axs = subplots(1, 1)

        # plot circle
        axs.plot(self.s21.real, self.s21.imag, '.-')
        # show f0
        axs.plot(self.hdr['IF0'], self.hdr['QF0'], 'rs')

        axs.set_xlabel('I [V]')
        axs.set_ylabel('Q [V]')
        axs.set_title('IQ circle')
        axs.axis('equal')
        axs.grid()