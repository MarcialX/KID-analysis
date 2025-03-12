# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. overdriven attenuation iteractive figure
# overdriven_figures.py
# Class to display interactive figure to handle overdriven resonators
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


import sys
import numpy as np

from scipy.signal import savgol_filter
from scipy import interpolate

from misc.funcs import *
from physics.funcs import *

from data_processing import *
from homodyne_functions import *



class overdrivenFigure:
    """ Interactive tool to set overdriven resonators. """

    def __init__(self, fig, axs, over_attens_by_temp, 
                 over_atts_mask, over_atts_mtx, number_fig_over, **kwargs):
        """
        Parameters
        ----------
        fig:        [figure]
        axs:        [axs] canvas axis.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Number of columns
        num_cols = kwargs.pop('num_cols', 5)
        # Number of rows
        num_rows = kwargs.pop('num_rows', 6)
        # ----------------------------------------------

        self.fig = fig
        self.axs = axs

        # plots dimensions
        self.num_cols = num_cols
        self.num_rows = num_rows

        self.over_attens_by_temp = over_attens_by_temp

        self.over_atts_mask = over_atts_mask
        self.over_atts_mtx = over_atts_mtx

        self.number_fig_over = number_fig_over

        self.onclick = self.fig.canvas.mpl_connect("button_press_event", self.onclick_sel_att_event)
        self.keyboard = self.fig.canvas.mpl_connect('key_press_event', self.key_sel_att_event)
        
        show()


    def onclick_sel_att_event(self, event):
        """ Subplot click event. """

        for i in range(self.num_rows):
        
            for j in range(self.num_cols):
        
                if event.inaxes == self.axs[i, j]:
        
                    if self.over_atts_mask[i, j]:
        
                        self.update_overdriven_plot(i, j)
                        self.over_attens_by_temp[self.number_fig_over*self.num_rows+i] = self.over_atts_mtx[i, j]


    def key_sel_att_event(self, event):
        """ Subplot keyboard event. """ 
        
        sys.stdout.flush()
        
        if event.key in ['x', 'd']:

            if event.key == 'x':
                
                self.fig.canvas.mpl_disconnect(self.onclick)
                self.fig.canvas.mpl_disconnect(self.keyboard)
                close(self.fig)

                printc(f'Changes saved!', 'info')

            elif event.key == 'd':

                for i in range(6):
                    self.update_overdriven_plot(i, 2)
                    self.over_attens_by_temp[self.number_fig_over*self.num_rows+i] = self.over_atts_mtx[i, 2]

                printc(f'Undoing changes', 'info')


    def update_overdriven_plot(self, i, j):
        """
        Update overdriven plot.
        Parameters
        ----------
        i,j:        [int,int] subplot position.
        ----------
        """
        
        for m in range(self.num_cols):

            if self.over_atts_mask[i, m]:
                if m == j:
                    self.axs[i, m].patch.set_facecolor('green')
                else:
                    self.axs[i, m].patch.set_facecolor('blue')
                
                self.axs[i, m].patch.set_alpha(0.2)
        
        self.fig.canvas.draw_idle()