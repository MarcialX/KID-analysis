# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. Data processing functions
# data_processing.py
# Diverse functions to data processing.
#
# Marcial Becerril, @ 11 February 2025
# Latest Revision: 11 Feb 2025, 17:03 UTC
#
# TODO list:
# Functions missing:
#	+ Savgol filter
#	+ PCA filter
#	+ High/low-pass filter
#	+ Stop band filters (60 Hz signal, PTC, etc.)
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


import numpy as np
import time
import random

from scipy import signal
from scipy.signal import butter, lfilter

from matplotlib.pyplot import *

from misc.funcs import *


# F U N C T I O N S
# ---------------------------------

# despiking tool
# -------------------------------
def glitch_filter(stream, win_size=100, sigma_thresh=4, source_min_size=None, 
				  win_noise_sample=0.25, trim_glitch_sample=3, verbose=True):
	"""
	Glitch filter
	Remove glitches and likely cosmic ray events.
	Parameters
	----------
	stream:				[array] time stream array.
	win_size:			[int] window size for local inspection.
	sigma_thresh:		[float] sigma threshold.
	source_min_size:	[array] definition of the minimum number of points of source.
						If None, it is equal to win_size.
	win_noise_sample:	[float] fraction of window size at the edges to define the noise sample
						which stats will replace a glitch event.
	trim_glitch_sample: [int] number of points to average to replace a single glitch event.
	verbose:			[bool] show some verbose.
	----------
	"""

	if source_min_size == None:
		source_pts = win_size
	else:
		source_pts = source_min_size

	deglitched_data = np.copy(stream)

	prev_cr_pts = np.array([])

	start_time = time.time()
	check_time = time.time()

	while True:

		data_win_diff = np.gradient(deglitched_data)    # get differentiated data
		sigma_diff = np.nanstd(data_win_diff)       	# dispersion of differentiated data
		offset_diff = np.nanmedian(data_win_diff)     	# average of differentiated data

		# cosmic ray events
		cr_idx = np.where((data_win_diff > offset_diff+sigma_thresh*sigma_diff) |
						   (data_win_diff < offset_diff-sigma_thresh*sigma_diff) )[0]

		# cosmic ray mask, have the cosmic rays been repaired?
		cr_mask = np.zeros_like(cr_idx, dtype=bool)

		num_cr_pts = len(cr_idx)

        # break loop conditions
		if check_time - start_time > 10:    # if it takes too much time or non-convergence
			break

		if num_cr_pts == 0 or np.array_equal(prev_cr_pts, cr_idx): 
			break

		if verbose:
			printc(f'Cosmic ray events: {num_cr_pts}', 'ok')

		# Get statistics per each point
		for e, cr in enumerate(cr_idx):

			if not cr_mask[e]:

				data_win = deglitched_data[cr-int(win_size/2):cr+int(win_size/2)]

				edge_data = np.concatenate((data_win[:int(win_noise_sample*win_size)], data_win[-1*int(win_noise_sample*win_size):]))

				if len(edge_data) > 0:

					sigma = np.std(edge_data)   # dispersion in selected window
					offset = np.mean(edge_data) # average in selected window

					# validate that it is a glitch event
					cr_rec = np.where((data_win > offset+sigma_thresh*sigma) |
									(data_win < offset-sigma_thresh*sigma) )[0]

					diff_cr_rec = np.diff(cr_rec)

					# replace glitch by random data with stats similar as non-glitched data
					if (np.count_nonzero(diff_cr_rec == 1) <= source_pts):

						# new random points with normal distribution
						random_sample = np.random.normal(offset, sigma, win_size)

						new_sample = np.zeros(len(diff_cr_rec)+1)
						for i in range(len(new_sample)):
							idx_sample = random.randint(0, win_size-1)
							new_sample[i] = random_sample[idx_sample]

						# update points
						idx_in_full_array = cr_rec + cr - int(win_size/2)
						deglitched_data[idx_in_full_array] = new_sample

						for m, idx in enumerate(cr_idx):
							if idx in idx_in_full_array:
								cr_mask[m] = True	

					else:
						deglitched_data[cr] = np.mean(np.concatenate((deglitched_data[cr-trim_glitch_sample:cr], \
													deglitched_data[cr+1:cr+trim_glitch_sample+1])))
						
						# glitch corrected
						cr_mask[e] = True
		
		check_time = time.time()    # get computation time of every cycle

		prev_cr_pts = cr_idx

	return deglitched_data


# binning
# -------------------------------
def log_binning(frequency, signal, n_pts=500):
	"""
	Logarithmic binning for PSD.
	Parameters
	-----------
	frequency:		[array] frequency [Hz].
	signal:			[array] signal.
	n_pts:			[int] number of points.
	-----------
	"""

	# set the limits
	start = frequency[0]
	stop = frequency[-1]

	# central positions of new frequency array
	central_freqs = np.logspace(np.log10(start), np.log10(stop), n_pts+1)

	n_freq = []
	n_signal = []
	for i in range(n_pts):
		# get the samples
		idx_start = np.where(frequency > central_freqs[i])[0][0]
		idx_stop = np.where(frequency <= central_freqs[i+1])[0][-1] + 1

		if len(frequency[idx_start:idx_stop]) > 0:
			if not np.isnan(np.mean(frequency[idx_start:idx_stop])):
				# get average points
				n_freq.append(np.mean(frequency[idx_start:idx_stop]))
				n_signal.append(np.median(signal[idx_start:idx_stop]))

	n_freq = np.array(n_freq)
	n_signal = np.array(n_signal)

	return n_freq, n_signal


def lin_binning(frequency, signal, w=10):
	"""
	Linear binning PSD.
	Parameters
	-----------
	frequency:		[array] frequency array [Hz].
	signal:			[array] signal.
	w:				[int] size binning.
	-----------
	"""

	frequency_accum, signal_accum = 0, 0

	n_signal = []
	n_freq = []
	for i, p in enumerate(signal):
		
		if i%w == 0 and i != 0:
			# append new points
			n_signal.append(signal_accum/w)
			n_freq.append(frequency_accum/w)
			
			# restart accumulators
			signal_accum = 0
			freq_accum = 0

		signal_accum += p
		freq_accum += frequency[i]

	n_freq = np.array(n_freq)
	n_signal = np.array(n_signal)

	return n_freq, n_signal


# binning and merging
# -------------------------------
def merge_spectra(frequencies, signals, n_pts=500):
	"""
	Merge two spectra.
	Parameters
	----------
	frequencies:		[2d-array] frequency arrays.
	signals:			[2d-array] signal arrays to merge.
	n_pts:				[int] len in points of the merge data.
	----------
	"""

	nsamples = len(frequencies)

	# get frequency limits
	mins, maxs = [], []
	for n in range(nsamples):
		mins.append(np.min(frequencies[n]))
		maxs.append(np.max(frequencies[n]))

	# set the limits
	start = np.min(mins)
	stop = np.max(maxs)

	# central positions of new frequency array
	central_freqs = np.logspace(np.log10(start), np.log10(stop), n_pts+1)

	n_frequency = []
	n_signal = []

	for i in range(n_pts):

		accum_points = np.array([])		# define accumulators
		accum_freqs = np.array([])

		# sample all the frequencies
		for n in range(nsamples):
			# get the samples

			lower_idx = np.where(frequencies[n] >= central_freqs[i])[0]
			if len(lower_idx) > 0:
				idx_start = lower_idx[0]
			else:
				idx_start = None
			
			upper_idx = np.where(frequencies[n] < central_freqs[i+1])[0]
			if len(upper_idx) > 0:
				idx_stop = upper_idx[-1] + 1
			else:
				idx_stop = None

			#print(idx_start, idx_stop)

			if idx_start != None and idx_stop != None:
				accum_points = np.concatenate((accum_points, signals[n][idx_start:idx_stop]))
				accum_freqs = np.concatenate((accum_freqs, frequencies[n][idx_start:idx_stop]))

		if len(accum_points) > 0:
			n_signal.append(np.mean(accum_points))
			n_frequency.append(np.mean(accum_freqs))

	return np.array(n_frequency), np.array(n_signal)


# filters
# -------------------------------
def butter_hipass_model(cutoff, fs, order=5):
	"""
	Butterworth high-pass model.
	Parameters
	----------
	cutoff:		[float] frequency cutoff.
	fs:			[float] sampling frequency.
	order:		[int] filter order.
	----------
	"""

	nyq = 0.5 * fs		# Nyquist frequency
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)

	return b, a


def butter_hipass_filter(data, cutoff, fs, order=5):
	"""
	Apply butterworth high-pass filter.
	Parameters
	----------
	data:		[array] raw data.
	cutoff:		[float] frequency cutoff.
	fs:			[float] sampling frequency.
	order:		[int] filter order.
	----------
	"""

	b, a = butter_hipass_model(cutoff, fs, order=order)
	return lfilter(b, a, data)


def sinc_lopass_model(fc=0.1, b=0.08):
	"""
	Windowed-sinc low-pass model.
	This code is not mine, but I don't remember from where I got it.
	Parameters
	----------
	fc:		[float] frequency cutoff.
	b:		[float] b parameter.
	----------
	"""

	N = int(np.ceil((4 / b)))

	if not N % 2: N += 1  # make sure that N is odd.
	n = np.arange(N)

	# compute sinc filter.
	h = np.sinc(2 * fc * (n - (N - 1) / 2))

	# compute Blackman window.
	w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
	    0.08 * np.cos(4 * np.pi * n / (N - 1))

	# multiply sinc filter by window.
	h = h * w

	# normalize to get unity gain.
	h = h / np.sum(h)

	return h


def sinc_lopass_filter(data, fc, b):
	"""
	Apply sinc low pass filter.
	Parameters
	----------
	data:	[array] raw data array.
	fc:		[float] frequency cutoff.
	b:		[float] b parameter.
	----------
	"""
	
	h = sinc_lopass_model(fc, b)	# get sinc filter model
	
	filter_data = np.convolve(data, h)
	filter_data = filter_data[int((len(h)-1)/2):-int((len(h)-1)/2)]
	
	return filter_data


def notch_model(fs, f0, Q):
	"""
	Notch filter model.
	Parameters
	------------
	fs:		[float] sample frequency.
	f0:		[float] frequency to filter.
	Q:		[float] quality factor.
	----------
	"""

	b, a = signal.iirnotch(f0, Q, fs)
	return b, a


def notch_filter(data, fs, f0, Q):
	"""
	Apply notch filter.
	Parameters
	------------
	fs:		[float] sample frequency.
	f0:		[float] frequency to filter.
	Q:		[float] quality factor
	------------
	"""

	b, a = notch_model(fs, f0, Q)
	filter_data = signal.filtfilt(b, a, data)
	
	return filter_data
