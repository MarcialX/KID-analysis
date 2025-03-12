# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. noise measurements class
# noise_meas.py
# Class to handle noise measurements.
#
# Marcial Becerril, @ 11 February 2025
# Latest Revision: 11 Feb 2025, 16:24 UTC
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

from data_processing import *
from homodyne_functions import *



# N O I S E   C L A S S
# -----------------------------------
class noise:
    """ s21 measurements class. """

    def __init__(self, I, Q, hdr, mask_samples=[]):
        """
        Parameters
        ----------
        I/Q:        [2d-array] I/Q timestream data.
        hdr:        [dict] header data.
        ----------
        """

        # check I, Q data format
        assert isinstance(I, list) or isinstance(I, np.ndarray), \
            "I data format not valid."
        
        assert isinstance(Q, list) or isinstance(Q, np.ndarray), \
            "Q data format not valid."

        self.I, self.Q = I, Q       # assign I, Q data
        self.samples = len(I)       # get the number of samples

        # generate the mask for the timestreams
        mask = np.ones(self.samples, dtype=bool)
        for sample in mask_samples:
            try:
                mask[sample] = False

            except IndexError:
                printc(f"Sample {sample} doesn't exist.", "warn")

        self.mask = mask

        # initialise header data
        self.hdr = hdr

        # get time array
        time_len = self.hdr['SAMPLELE']
        self.time_sample = np.linspace(0, time_len, len(I[0]))
        self.hdr['TIMESTEP'] = np.mean(np.diff(self.time_sample))

        # initialise df
        self.df = None

        # ... and psd
        self.psd = None 


    def despike(self, active_mask=True, **kwargs):
        """
        Apply despiking filter to selected samples.
        Parameters
        ----------
        active_mask:    [bool] activate mask.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        #time = self.time_sample       # <--- MIGHT BE USELESS

        if active_mask:
            samples_to_use = np.arange(self.samples)[self.mask]
        else:
            samples_to_use = np.arange(self.samples)

        for sample in samples_to_use:

            if verbose:
                printc(f'Cleaning sample: {sample}', 'info')
            
            self.I[sample] = glitch_filter(self.I[sample], verbose=verbose, **kwargs)
            self.Q[sample] = glitch_filter(self.Q[sample], verbose=verbose, **kwargs)


    def auto_masking(self, sigma_thresh=2.85, freq_avg_psd=[4, 20], alarm=4, show_masked=True):
        """
        Auto mask timestream samples based on the consistency 
        of dispersion.
        Parameters
        ----------
        sigma_thresh:       [float] sigma threshold.
        ----------
        """

        assert isinstance(alarm, int), "The alarm has to be an integer number"

        # get sampling frequency
        fs = self.hdr['SAMPLERA']

        # average holders
        avg_psd_I, avg_psd_Q = [], []

        if self.samples > 2:

            # get a quick psd of each sample
            for sample in range(self.samples):
                # compute psd
                f_I, psd_I = get_psd(self.I[sample], fs)
                _,   psd_Q = get_psd(self.Q[sample], fs)

                # get averaged psd
                #from_freq_idx = np.where(f_I>=freq_avg_psd[0])[0][0]
                #to_freq_idx = np.where(f_I>=freq_avg_psd[1])[0][0]

                avg_psd_I.append(np.mean(psd_I[freq_avg_psd[0]:freq_avg_psd[1]]))
                avg_psd_Q.append(np.mean(psd_Q[freq_avg_psd[0]:freq_avg_psd[1]]))

            # get mean of average psd levels
            med_avg_psd_I = np.median(avg_psd_I)
            med_avg_psd_Q = np.median(avg_psd_Q)

            std_avg_psd_I = np.std(avg_psd_I)
            std_avg_psd_Q = np.std(avg_psd_Q)    

            # select timestreams
            select_I = (avg_psd_I < (med_avg_psd_I+sigma_thresh*std_avg_psd_I)) & \
                    (avg_psd_I > (med_avg_psd_I-sigma_thresh*std_avg_psd_I))

            select_Q = (avg_psd_Q < (med_avg_psd_Q+sigma_thresh*std_avg_psd_Q)) & \
                    (avg_psd_Q > (med_avg_psd_Q-sigma_thresh*std_avg_psd_Q))
            
            preliminar_mask = select_I & select_Q

            # get number of masked samples
            num_masked_samples = len(preliminar_mask) - np.count_nonzero(preliminar_mask)
            printc(f"Number of masked samples: {num_masked_samples}", "info")
            if num_masked_samples > alarm:
                printc(f"Excess of masked samples", "warn")

            # print masked samples
            masked_samples = ""
            for i in range(len(preliminar_mask)):
                if not preliminar_mask[i]:
                    masked_samples += f"{i},"
            printc(f"Masked samples: {masked_samples[:-1]}", "warn")

            print(preliminar_mask)

            # show masked samples?
            if show_masked:
                for i, mask in enumerate(preliminar_mask):
                    if not mask:
                        fig, axs = subplots(2, 1, sharex=True)
                        axs[0].plot(self.time_sample, self.I[i], lw=1)
                        axs[1].plot(self.time_sample, self.Q[i], lw=1)
                        axs[0].set_title(f'Sample: {i}')
                        axs[1].set_xlabel(f'Time [s]')
                        axs[0].set_ylabel(f'I[V]')
                        axs[1].set_ylabel(f'Q[V]')

            self.mask = preliminar_mask

        else:
            printc(f'Auto masking only operates with at least three samples.', 'warn')


    @property
    def mean_samples(self):
        """ Get average. """
        return np.mean(self.I, axis=1), np.mean(self.Q, axis=1)


    @property
    def median_samples(self):
        """ Get medians. """
        return np.median(self.I, axis=1), np.median(self.Q, axis=1)
    

    @property
    def stdev_samples(self):
        """ Get standard deviation. """
        return np.std(self.I, axis=1), np.std(self.Q, axis=1)

# Parse cosmic rays events extraction