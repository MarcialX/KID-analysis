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


# iq circle tools
# -------------------------------
def fit_circ(x, y):
    """
    This code belongs to Andreas Papagiorgou.
    Get center and radius of a circle from its circunference.
    Parameters
    ----------
    x, y:   [array] x/y data.
    ----------
    """

    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # calculate distances from centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1 - R_1)**2)

    return xc_1, yc_1, R_1


def derot_circle(Is, Qs, xc, yc, theta):
    """
    Rotate and translate array/point.
    Parameters
    ----------
    Is/Qs:      [arrays] Is/Qs resonance frequency sweeps.
    xc/yc:      [float/float] circle center.
    theta:      [float] rotation angle.
    ----------
    """

    Is_derot = (Is - xc)*np.cos(-theta)-(Qs - yc)*np.sin(-theta)
    Qs_derot = (Is - xc)*np.sin(-theta)+(Qs - yc)*np.cos(-theta)

    return Is_derot, Qs_derot


def get_phase(s21, phase_offset=0):
	"""
	Get phase from s21 complex array.
	Parameters
	----------
	s21:			[array/float] s21 data as complex number.
	phase_offset:	[float] phase offset.
	----------
	"""

	I, Q = s21.real, s21.imag               # get I,Q components.

	phase = np.unwrap(np.arctan2(Q, I))     # get unwrapped phase.

	# correct phase
	if np.min(phase) > 0:
		phase = -2*np.pi + phase

	elif np.max(phase) < 0:
		phase = phase + 2*np.pi

	return phase + phase_offset


def get_circle_speed(freq, s21):
	"""
	Get circle I/Q speed.
	Parameters
	----------
	freq:   [array] frequency array.
	s21:    [array] s21 data as complex array.
	----------
	"""

	I, Q = s21.real, s21.imag

	I_speed = np.gradient(I, freq)
	Q_speed = np.gradient(Q, freq)

	# compute speed
	return np.sqrt(I_speed**2 + Q_speed**2)
