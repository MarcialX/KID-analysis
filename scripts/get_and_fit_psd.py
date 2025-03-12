# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Script to get PSD of available noise data + fit noise
#
# Marcial Becerril, @ 11 March 2025
# Latest Revision: 11 Mar 2025, 14:07 UTC
#
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

sys.path.append('../')
from data_analyser import *

from matplotlib.pyplot import *
rc('font', family='serif', size='18')


# I N I T
# -------------------------------------
# meas properties
device = 'FT151'
chip_number = 1
cryostat = 'Eagle'
date = '250227'

# work directory
work_diry = '/home/marcial/Documents/FT151-ANL/Chip-1/'

# chip name
chip_name = f'FT151-{cryostat}-Chip{chip_number}-{date}'

# chip data
chip_path = f"/home/marcial/detdaq/Eagle/FT151_Chip_1"

# read data
lab = 'pete-lab'
if cryostat in ['Elmo', 'Aloysius', 'Animal']:
    lab = 'kids-lab'

# reference temperature for summary
ref_temp = 128

chip = KidAnalyser(chip_path, only_vna=False, lab=lab, meas_date=date, project_name=chip_name, work_dir=work_diry)

# extra input atten
chip.add_input_atten = 40

# check if overdriven defined
if len(chip.sweep_overdriven) == 0:
    # run the overdriven definition tool
    chip.find_overdriven_atts(ref_temp)
    # save overdriven attens
    chip.save_opt_attens()

# C O M P U T E   P S D
# -------------------------------------
# typical good psd params.
psd_params = {
    'df_method': 'phase-inter',
    'trim_psd_edges': [3, -500],
    'plot_phase':   True,
    'plot_streams': True,
    'despike':  True,
    'ffs_units': False,
    'binning_points':   400,
    'sweep_sample': 0,
    'mask':     True,
    'f0_phase_thresh': 15e4,
    'linear_fit':   True,
    'f0_thresh_linear_fit': 2e3,
    'smooth_params_phase': [5, 3]
}

# fit the psd on - off
psd_to_fit = 'on-off'

# kids to get noise
kids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
temperatures = [128]

for kid in kids:
    for temperature in temperatures:
        
        # get all the atts available
        k = chip._parse_single_item(kid, prefix='K', nzeros=3)                                          # get kid
        t = chip._parse_single_item(temperature, prefix=chip._temp_prefix, nzeros=chip._temp_zeros)     # get temperature
        attenuations = list(chip.data['hmd'][k][t].keys())
        
        for attenuation in attenuations:
            
            printc(f'Processing {k}-{t}-{attenuation}...', 'info')

            # get psd and fit
            f, psds, popt, fitted_psd = chip.on_off_psd_resonance(kid, temperature, attenuation, 
                                                            samples_onr="all", samples_off="all",
                                                            psd_params=psd_params, psd_to_fit=psd_to_fit)
            
            printc(f'Done', 'ok')

