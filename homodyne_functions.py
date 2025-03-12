# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. Power spectral density functions.
# psd_functions.py
# Functions to process power spectra fucntions
#
# Marcial Becerril, @ 11 February 2025
# Latest Revision: 11 Feb 2025, 17:03 UTC
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import os
import numpy as np

from scipy import signal

from matplotlib.pyplot import *

from misc.funcs import *
from data_processing import *
from sweep_functions import *

# F U N C T I O N S
# ---------------------------------
def get_homodyne_files(directory):
    """
    Get timestream files.
    Parameters
    ----------
    directory:      [str] homodyne directory.
    ----------
    """

    # get files
    files = os.listdir(directory)

    # sweep data
    sweep_path = []
    sweep_hr_path = []

    # noise paths
    noise_on_path = []
    noise_off_path = []

    # check all the files
    for file in files:
        # check a single file
        k = file.lower()

        if "sweep" in k:
            file_no_ext = file.split('.')[0]
            split_sweep = file_no_ext.split("_")[-1]

            if split_sweep[-1].isnumeric():
                sweep_sample = int(split_sweep[-1])
            else:
                sweep_sample = 0

            if "hr" in k:
                sweep_hr_path.append([file, sweep_sample])
            else:
                sweep_path.append([file, sweep_sample])

        else:
            fs, mode, n, cnt = "", "", "", 0
            for j in k:
                if j == "_":
                    cnt += 1
                elif j == ".":
                    cnt = 0
                elif cnt == 1:
                    fs = fs + j
                elif cnt == 3:
                    mode = mode + j
                elif cnt == 6:
                    n = n + j
            
            if n == "":
                n = 0

            if mode == "on":
                noise_on_path.append([file, int(fs), int(n)])

            elif mode == "off":
                noise_off_path.append([file, int(fs), int(n)])

    return sweep_path, sweep_hr_path, noise_on_path, noise_off_path


def df_from_magic(I, Q, didf, dqdf, I0, Q0):
    """
    Get the resonance frequency shift through the "magic" formula.
    [space for the reference].
    Parameters
    ----------
    I/Q:        [array] I/Q timestream.
    didf/dqdf:  [float] I/Q sweep gradient.
    I0/Q0:      [float] I0/Q0 at the resonance frequency.
    ----------
    """

    # get dxdf magnitude
    dIQ = didf**2 + dqdf**2
    # get df
    df = ( ((I - I0)*didf) + ((Q - Q0)*dqdf) ) / dIQ

    return df


def df_from_phase(s21, f0, phase_model, phase_offset=0):
    """
    Get df from phase through interpolation.
    Parameters
    ----------
    s21:        [array] s21 data array.
                It has to be derotated.
    f0:         [float] resonance frequency.
    f0_model:   [object] f0_model(phase) model.
    phase_offset:  [float] phase offset.
    ----------
    """

    # get phase
    phase = get_phase(s21)

    # get df
    df = phase_model(phase - phase_offset) - f0
    
    return df


def get_psd(df, fs):
    """
    Compute the power spectral density (psd) through the periodogram method.
    Parameters
    ----------
    df:             [array] resonance frequency shift [Hz].
    fs:             [float] sampling frequency [Hz].
    ----------
    """

    return signal.periodogram(df, fs)

