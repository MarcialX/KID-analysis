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

import sys
import numpy as np

#sys.path.append('../')
from misc.funcs import *
from misc.timeout import timeout

from fit_resonator_model.resonator_models import *



@timeout(150)
def fit_single_resonator(frequency, s21, n=3.5, tau=50e-9):
    """
    Fit resonator based on Gao model.
    Parameters
    ----------
    frequency:      [array] frequency [Hz].
    s21:            [array] sweep complex data.
    n[opt]:         [float] number of linewdiths to extract data.
    tau[opt]:       [float] cable delay [s].
    ----------
    """

    # instance resonator model
    res_model = ResGaoModel(tau=tau)

    # guess initial parameters
    guess = res_model.guess(frequency, s21)

    # linewidth
    lw = guess['rgm_fr'].value/np.abs(guess['rgm_Qr'].value)

    # get f0 index based on the min(s21)
    m_idx = np.argmin(np.abs(s21))

    from_idx = frequency[m_idx]-n*lw
    if from_idx < frequency[0]:
        from_idx = 0
    else:
        from_idx = np.where(frequency>=from_idx)[0][0]

    to_idx = frequency[m_idx]+n*lw
    if to_idx > frequency[-1]:
        to_idx = len(frequency)
    else:
        to_idx = np.where(frequency>=to_idx)[0][0]

    guess = res_model.guess(frequency[from_idx:to_idx], s21[from_idx:to_idx])

    # show the initial parameters
    printc(f'I N I T I A L   P A R A M E T E R S', 'title3')
    printc(f'fr: {1e-6*guess['rgm_fr'].value:,.3f} MHz', 'info')
    printc(f'Qr: {guess['rgm_Qr'].value:,.0f}', 'info')
    printc(f'Qc: {guess['rgm_Qc'].value:,.0f}', 'info')
    Qi_guess = guess['rgm_Qr'].value*guess['rgm_Qc'].value / (guess['rgm_Qc'].value - guess['rgm_Qr'].value)
    printc(f'Qi: {Qi_guess:,.0f}', 'info')
    printc(f'tau: {1e9*tau:.2f} [ns]', 'info')

    printc(f'Segment selected: {from_idx}-{to_idx}, total points: {to_idx-from_idx}', 'info')

    result = res_model.fit(s21[from_idx:to_idx], params=guess, frequency=frequency[from_idx:to_idx])
    #fit_s21 = res_model.eval(params=result.params, frequency=frequency)

    # get fit results
    ar = result.values['rgm_ar']
    ai = result.values['rgm_ai']
    fr = result.values['rgm_fr']
    Qr = result.values['rgm_Qr']
    Qc = result.values['rgm_Qc']
    phi = result.values['rgm_phi']
    non = result.values['rgm_non']

    # compute Qi
    Qi = get_Qi(Qr, Qc)

    fit_s21 = resonator_model(frequency, fr, ar, ai, Qr, Qc, phi, non, tau=tau)

    # and Qi error
    try:
        Qr_err = result.uvars['rgm_Qr'].std_dev
        Qc_err = result.uvars['rgm_Qr'].std_dev

        Qi_err = get_Qi_err(Qr, Qc, Qr_err, Qc_err)
        printc(f'Qi error: {Qi_err:.0f}', 'info')

    except Exception as e:
        Qi_err = None
        printc(f'Error computing Qi error.\n{e}', 'warn')

    # save fit results as a dictionary
    fit_kid = {}

    fit_kid['ar'] = ar
    fit_kid['ai'] = ai
    fit_kid['fr'] = fr
    fit_kid['Qr'] = Qr
    fit_kid['Qc'] = Qc
    fit_kid['Qi'] = Qi
    fit_kid['phi'] = phi
    fit_kid['non'] = non
    
    # get error per parameter
    try:
        fit_kid['ar_err'] = result.uvars['rgm_ar'].std_dev
        fit_kid['ai_err'] = result.uvars['rgm_ai'].std_dev
        fit_kid['fr_err'] = result.uvars['rgm_fr'].std_dev
        fit_kid['Qr_err'] = result.uvars['rgm_Qr'].std_dev
        fit_kid['Qc_err'] = result.uvars['rgm_Qc'].std_dev
        fit_kid['phi_err'] = result.uvars['rgm_phi'].std_dev
        fit_kid['non_err'] = result.uvars['rgm_non'].std_dev
        fit_kid['Qi_err'] = Qi_err

    except:
        pass

    fit_kid['fit_sweep'] = fit_s21

    return fit_kid


def get_Qi_err(Qr, Qc, Qr_err, Qc_err):
    """ Get Qi error from error propagation. """

    e1 = -Qr**2/(Qc-Qr)**2
    e2 = Qc**2/(Qc-Qr)**2

    # get Qi from error propagation
    Qi_err = np.sqrt( (e1*Qc_err)**2 + (e2*Qr_err)**2 )

    return Qi_err


def get_Qi(Qr, Qc):
    """ Get Qi from Qr and Qc. """

    return Qr*Qc / (Qc - Qr)
