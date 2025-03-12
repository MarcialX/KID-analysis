#! /home/marcial/.venv/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID finder tool.
# kids_finder.py
#
# Marcial Becerril, @ 16 Aug 2024
# Latest Revision: 16 Aug 2024, 13:05 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import sys
import argparse
import numpy as np

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import interpolate

from matplotlib.pyplot import *

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.family'] = 'serif'

sys.path.append('../')
from fit_resonator_model.resonator_models import *


# F U N C T I O N S
# --------------------------------------------------
def find_kids(frequency, s21, down_factor=35, baseline_params=(501, 5), 
              Qr_lim=[1500, 150000], Qc_lim=[1000, 150000], inter=True, **kwargs):
    """
    Find resonators from the VNA sweep.
    Parameters
    ----------
    frequency:          [array] frequency array [Hz].
    s21:                [array] S21 complex array.
    down_factor:        [int] downsampling factor to smooth data and extract baseline.
    baseline_params:    [tuple] savinsky-golay filter:
                        idx 0: number of points.
                        idx 1: order.
    Qr_lim:             [list] Qr upper(1) and lower(0) limits. 
    Qc_lim:             [list] Qc upper(1) and lower(0) limits.
    inter:              [bool] interactive mode. 
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Savinsky-Golay parameters
    #   -> number of points
    base_smooth_points = kwargs.pop('base_smooth_points', 7)
    #   -> order
    base_smooth_order = kwargs.pop('base_smooth_order', 2)
    # cable delay
    tau = kwargs.pop('tau', 50e-9)
    # Peaks distance
    distance = kwargs.pop('distance', 50)
    # Peaks height
    height = kwargs.pop('height', 0.5)
    # Peaks prominence
    prominence = kwargs.pop('prominence', 0.5)
    # ----------------------------------------------

    # invert the S21 sweep
    # ---------------------------------
    s21_log = 20*np.log10(np.abs(s21))
    s21_log_inv = -1*s21_log
    
    # downsampling
    # ---------------------------------
    s21_down = s21_log_inv[::down_factor]
    frequency_down = frequency[::down_factor]

    s21_down = np.append(s21_down, s21_log_inv[-1])
    frequency_down = np.append(frequency_down, frequency[-1])

    # extract the baseline
    # ---------------------------------
    npoints = baseline_params[0]
    order = baseline_params[1]

    baseline_down = savgol_filter(s21_down, npoints, order)
    inter_mdl = interpolate.interp1d(frequency_down, baseline_down)
    baseline = inter_mdl(frequency)

    s21_no_base = s21_log_inv - baseline

    sm_s21_no_base = savgol_filter(s21_no_base, base_smooth_points, base_smooth_order)
    peaks, _ = find_peaks(sm_s21_no_base, distance=distance, height=height, prominence=prominence)

    df = np.mean(np.diff(frequency))
    nw_peaks = []
    flags = []
    for peak in peaks:

        size_sample = 6*frequency[peak]/Qr_lim[0]/2
        range_a = peak - int(size_sample/df)
        range_b = peak + int(size_sample/df)

        if range_a < 0:
            range_a = 0
        if range_b > len(frequency):
            range_b = len(frequency)-1

        # take a sample
        freq_sm = frequency[range_a:range_b]
        s21_sm = s21[range_a:range_b]

        try:
            ar, ai, Qr, fr, Qc, phi = guess_resonator_params(freq_sm, s21_sm, tau=tau)

            if (Qr > Qr_lim[0] and Qr < Qr_lim[1]) and (Qc > Qc_lim[0] and Qc < Qc_lim[1]):
                flags.append(True)
            else:
                flags.append(False)

            nw_peaks.append(peak)

        except Exception as e:
            printc(f'Peak: {peak} ignored.\n{e}', 'warn')

    # interactive mode
    if inter:
        interFinder = kidsFinderApp(frequency, s21, nw_peaks, flags)
        return interFinder.found_kids

    else:
        sel_peaks = []
        for p, peak in enumerate(nw_peaks):
            if flags[p]:
                sel_peaks.append(frequency[peak])
    
        return sel_peaks


class kidsFinderApp:
    """ Interactive tool to find resonators. """

    def __init__(self, frequency, s21, peaks, flags, **kwargs):
        """
        Parameters
        ----------
        frequency:      [array] frequency array.
        s21:            [array] sweep data.
        peaks:          [list] list of potential resoantors.
        flags:          []
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # frequency distance threshold
        freq_thresh = kwargs.pop('freq_thresh', 5e4)
        # ----------------------------------------------

        # original raw data
        self.frequency = frequency
        self.s21 = s21
        self.peaks = peaks
        self.flags = flags

        # backup data
        self.frequency_backup = np.copy(frequency)
        self.s21_backup = np.copy(s21)
        self.peaks_backup = np.copy(peaks)
        self.flags_backup = np.copy(flags)

        # peak frequency threshold
        self.freq_thresh = freq_thresh

        self.edit_mode = False
        self.range_flag = False
        self.range = [0, 0]
        self.found_kids = []

        # call the interactive tool
        self.init_interactive_tool(frequency, peaks, flags)


    def init_interactive_tool(self, frequency, peaks, flags):
        """
        Interactive mode to clean psd data.
        Parameters
        ----------
        frequency:      [array] frequency array [Hz].
        peaks:          [list] s21 peaks, potential resonators.
        flags:          [list] masked peaks.
        ----------
        """

        # create figure
        self.fig = figure()
        self.axs = self.fig.add_subplot(111)

        self.mode = True

        # initial plot
        self.flag_update =  True
        self.update_plot_find_kids(frequency, peaks, flags)

        self.onclick_xy = self.fig.canvas.mpl_connect('button_press_event', self.onclick_find_kids)
        self.keyboard = self.fig.canvas.mpl_connect('key_press_event', self.key_pressed_find_kids)

        show()


    def update_plot_find_kids(self, frequency, peaks, flags):
        """
        Update finding KIDs plot.
        Parameters
        ----------
        frequency:      [array] frequency array.
        peaks:          [list] s21 peaks, potential resonators.
        flags:          [list] masked peaks.
        ----------
        """

        # plot full sweep
        if self.flag_update:
            self.axs.plot(self.frequency, 20*np.log10(np.abs(self.s21)), 'b')

            instructions_text = "x : save and quit\nq : quit\nd : default\nu : update\ne :edit mode\n  a : add resonances\n  r : remove resonances\n    w : select range"
            self.axs.text(0.03, 0.05, instructions_text, fontsize=13, transform=self.axs.transAxes)

        # show found kids
        for p, peak in enumerate(peaks):
            if flags[p]:
                self.axs.axvline(frequency[peak], color='r')
            else:
                self.axs.axvline(frequency[peak], color='k', lw=0.35)

        if self.mode:
            self.axs.patch.set_facecolor('green')
            self.axs.patch.set_alpha(0.2)

        else:
            self.axs.patch.set_facecolor('red')
            self.axs.patch.set_alpha(0.2)

        self.axs.set_xlabel('Frequency [Hz]')
        self.axs.set_ylabel('S21 [dB]')

        if self.edit_mode:
            if self.mode:
                str_mode = "Add resonances"
            else:
                str_mode = "Remove resonances"

            self.axs.set_title(f'EDITION MODE. {str_mode}')

        else:
            self.axs.set_title('VISUALISATION MODE')

        # add a text box with the total number of found resonators
        n_kids = np.sum(self.flags)
        summary_text = f"Resonators : {n_kids}"
        self.axs.text(0.03, 0.95, summary_text, fontsize=16, transform=self.axs.transAxes)


    def key_pressed_find_kids(self, event):
        """
        Keyboard event to save/discard line fitting changes
        Keys:
            'x' : save and quit
            'q' : quit
            'd' : go back to default settings
            'u' : update
            'a' : add resonances
            'r' : remove resonances
            'e' : edit plot
            'z' : add/remove a tonelist where the cursor is
        """

        sys.stdout.flush()

        if event.key in ['x', 'q', 'd', 'u', 'a', 'r', 'e', 'w', 'z']:

            # save and quit
            if event.key == 'x':

                self.flag_update = False

                self.fig.canvas.mpl_disconnect(self.onclick_xy)
                self.fig.canvas.mpl_disconnect(self.keyboard)
                close(self.fig)

                # save data
                sel_peaks = []
                for p, peak in enumerate(self.peaks):
                    if self.flags[p]:
                        sel_peaks.append(self.frequency[peak])

                self.found_kids = sel_peaks

                # create tonelist
                sort_tones = np.sort(self.found_kids)

                tonelist_name = f'Toneslist-A.txt'
                with open(tonelist_name, 'w') as file:
                    file.write('Name\tFreq\tOffset att\tAll\tNone\n')
                    for kid in range(len(sort_tones)):
                        kid_name = 'K'+str(kid).zfill(3)
                        file.write(f'{kid_name}\t{sort_tones[kid]:.0f}\t0\t1\t0\n')
                
                printc(f'Tonelist file save as: {tonelist_name}', 'info')
                printc(f'Changes saved!', 'info')

            # update canvas
            elif event.key == 'u':
                
                cla()
                self.flag_update = True
                self.update_plot_find_kids(self.frequency, self.peaks, self.flags)
                self.fig.canvas.draw_idle()

            # add resonators mode activated
            elif event.key == 'a':

                self.mode = True
                self.flag_update = False
                self.range_flag = False

                self.update_plot_find_kids(self.frequency, self.peaks, self.flags)
                self.fig.canvas.draw_idle()

            # remove resonators mode activated
            elif event.key == 'r':
                
                self.mode = False
                self.flag_update = False
                self.range_flag = False

                self.update_plot_find_kids(self.frequency, self.peaks, self.flags)
                self.fig.canvas.draw_idle()
            
            # range selection
            elif event.key == 'w':
                
                self.flag_update = False
                self.range_flag = not self.range_flag

            # apply range removal
            elif event.key == 'z':

                self.flag_update = False
                self.range_flag = False

                upper_limit = np.max(self.range)
                lower_limit = np.min(self.range)

                for p, peak in enumerate(self.peaks):
                    fp = self.freq[peak]
                    if (fp > lower_limit) and (fp < upper_limit):
                        self.flags[p] = False

                self.update_plot_find_kids(self.frequency, self.peaks, self.flags)
                self.fig.canvas.draw_idle()              

            # activate edit mode
            elif event.key == 'e':
                
                self.edit_mode = not self.edit_mode
                
                if self.edit_mode:

                    if self.mode:
                        str_mode = 'Add resonances'
                    
                    else:
                        str_mode = 'Remove resonances'

                    self.axs.set_title('EDITION MODE. '+str_mode)
                    self.axs.tick_params(color='blue', labelcolor='blue')
                    
                    for spine in self.axs.spines.values():
                        spine.set_edgecolor('blue')

                else:
                    
                    self.range_flag = False
                    self.axs.set_title('VISUALISATION MODE')
                    self.axs.tick_params(color='black', labelcolor='black')
                    
                    for spine in self.axs.spines.values():
                        spine.set_edgecolor('black')

                self.fig.canvas.draw_idle()

            # go back to default mode
            elif event.key == 'd':

                self.frequency = self.frequency_backup
                self.s21 = self.s21_backup
                self.peaks = self.peaks_backup
                self.flags = self.flags_backup

                cla()
                self.update_plot_find_kids(self.frequency, self.peaks, self.flags)
                self.fig.canvas.draw_idle()
                printc(f'Undoing changes', 'info')


    def onclick_find_kids(self, event):
        """ On click event to select lines. """

        if event.inaxes == self.axs:

            # left-click
            if event.button == 3:
            
                ix, iy = event.xdata, event.ydata
                # add resonances
                flag_done = True
            
                for p, peak in enumerate(self.peaks):
            
                    if self.mode:
                        if np.abs(ix - self.frequency[peak]) < 1e3:
                            self.flags[p] = True 
                            flag_done = False
                            break

                    else:
                        if np.abs(ix - self.frequency[peak]) < self.freq_thresh:
                            self.flags[p] = False
                            flag_done = False
                            break

                if self.edit_mode:
                    
                    if flag_done and self.mode:
                        ix_idx = np.where(ix < self.frequency)[0][0]
                        self.peaks.append(ix_idx)
                        self.flags.append(True)

                    if self.mode:
                        self.axs.axvline(ix, color='g')

                    else:
                        if self.range_flag:
                            self.axs.axvline(ix, color='b', linestyle='dashed', linewidth=1.5)
                            self.range[1] = self.range[0]
                            self.range[0] = ix

                        else:
                            self.axs.axvline(ix, color='m', linestyle='dashed')

                    self.fig.canvas.draw_idle()

