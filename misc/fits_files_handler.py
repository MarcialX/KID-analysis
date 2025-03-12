# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID analysis. FITS files handler
# fits_files_handler.py
# FITS file handler: read, write, display
#
# Marcial Becerril, @ 09 February 2025
# Latest Revision: 09 Feb 2025, 18:45 UTC
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


from misc.funcs import *

from astropy.io import fits



# C L A S S E S
# -----------------------------------
class fitsHandler:
    """ FITS files handler """

    def load_fits(self, path, index=1):
        """
        Load FITS file given an existing path.
        Parameters
        ----------
        path:       [str] FITS file path to read.
        index:      [int] header/data index.
        ----------
        """

        assert isinstance(index, int), 'Index not valid, it has to be an integer number.'

        try:
            with fits.open(path) as hdul:
                hdr = hdul[index].header
                data = hdul[index].data

            printc(f"File {path} loaded!", "ok")
        
            return hdr, data

        except OSError as e:
            printc(f"{e}\nCheck if the file is an extesnion .fits or exists.", "fail")
        
        except IndexError as e:
            printc(f"{e}\nFITS file doesn't contain data the defined index.\n" + 
                   "Try another index or check if the FITS file is appropiate.", "fail")

        return 0


    def write_fits(self, data, hdr):
        """
        Write FITS file. 
        Parameters
        ----------
        data:   [array] data as a n-d array.
        hdr:    [dict] header data as a dictionary.
        ----------
        """
        pass