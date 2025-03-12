# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID analysis. Physical constants
# physical_constants.py
# Some useful physical constants
#
# Marcial Becerril, @ 09 February 2025
# Latest Revision: 09 Feb 2025, 22:45 UTC
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


Kb = 1.380649e-23       # Boltzmann constant [J K^-1]
h = 6.62607015e-34      # Planck constant [J Hz^-1]
c = 299792458           # light speed [m/s]
Tcmb = 2.725            # CMB temperature [K]

# single spin density of states at the Fermi level [um^-3 J^-1]
N0s = {
        'Al'    : 1.07e29,
        'AlTiAl': 1.07e29,      # from weightened average (ChatGPT): 1.00e29
        'AlMn': 1.07e29,        # no well-defined, assuming the Al
        'TiN'   : 2.43e29
}

# Tc [K]
Tcs = {
        'Al'    : 1.3,
        'AlTiAl': 1.01, 
        'AlMn'  : 0.85,         # no well-defined
        'TiN'   : 3.816         # Fb-Sp-TiN 35
}
