# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab.
# misc_functions.py
# Diverse functions
#
# Marcial Becerril, @ 28 May 2024
# Latest Revision: 28 May 2024, 12:50 GMT-6
#
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import sys
import numpy as np

from matplotlib.pyplot import *

from physics.physical_constants import *



def lin_binning(freq_psd, psd, w=10):
    """
    Linear binning PSD.
    Parameters
    -----------
    freq_psd : array
        Frequency [Hz].
    psd : array
        Power Spectral Density [Hz²/Hz].
    w : int
        Size binning.
    -----------
    """
    n_psd = []
    n_freq = []
    psd_accum = 0
    freq_accum = 0
    for i, p in enumerate(psd):
        if i%w == 0 and i != 0:
            n_psd.append(psd_accum/w)
            n_freq.append(freq_accum/w)
            psd_accum = 0
            freq_accum = 0
        psd_accum += p
        freq_accum += freq_psd[i]
    
    n_freq = np.array(n_freq)
    n_psd = np.array(n_psd)

    return n_freq, n_psd

def log_binning(freq_psd, psd, n_pts=500):
    """
    Logarithmic binning for PSD.
    Parameters
    -----------
    freq_psd : array
        Frequency [Hz].
    psd : array
        Power Spectral Density [Hz²/Hz].
    n_pts : int
        Number of points.
    -----------
    """

    start = freq_psd[0]
    stop = freq_psd[-1]

    central_pts = np.logspace(np.log10(start), np.log10(stop), n_pts+1)
    
    n_freq = []
    n_psd = []
    for i in range(n_pts):
        idx_start = np.where(freq_psd > central_pts[i])[0][0]
        idx_stop = np.where(freq_psd <= central_pts[i+1])[0][-1] + 1

        if not np.isnan(np.mean(freq_psd[idx_start:idx_stop])):
            n_freq.append(np.mean(freq_psd[idx_start:idx_stop]))
            n_psd.append(np.median(psd[idx_start:idx_stop]))

    n_freq = np.array(n_freq)
    n_psd = np.array(n_psd)

    return n_freq, n_psd

def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 500 random data sets are generated and fitted
    ps = []
    for i in range(1000):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    
    return pfit_bootstrap, perr_bootstrap 

