# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Script to fit resonators and create drive power vs Qi, Qc plots
#
# Marcial Becerril, @ 19 February 2025
# Latest Revision: 19 Feb 2025, 07:38 UTC
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


# F I T T I N G
# -------------------------------------
# check if overdriven defined
if len(chip.sweep_overdriven) == 0:
    # run the overdriven definition tool
    chip.find_overdriven_atts(ref_temp)
    # save overdriven attens
    chip.save_opt_attens()

# perform the fit
chip.fit_resonators(kids='all', overwrite=False, n=4)

# show all the sweeps
chip.plot_sweep_kids_vs_temps('all', attenuations='all', include_fit=False)


# F I T   R E S U L T S
# -------------------------------------
# show fit results

# flagged kids
bad_fit_res = []

# get summary
chip.qs_summary(temperature=ref_temp, flag_kids=bad_fit_res)

# display qs
chip.plot_power_vs_qs(temperatures="all", flag_kids=bad_fit_res)

show()

