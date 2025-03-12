# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. Data manager
# data_manager.py
# Class that manages files and data loading
#
# Marcial Becerril, @ 10 February 2025
# Latest Revision: 10 Feb 2025, 13:27 UTC
#
# TODO list:
#   - Basically everything ...
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


import os
from os import walk

import numpy as np
from tqdm import tqdm

from misc.funcs import *
from physics.funcs import *
from misc.fits_files_handler import *
from data_processing import *

from s21_meas import *
from noise_meas import *
from homodyne_functions import *



# M A N A G E R   C L A S S
# -----------------------------------
class DataManager(fitsHandler):
    """ Data manager class. """

    def __init__(self, directory, lab='pete-lab', **kwargs):
        """
        Parameters
        ----------
        directory:  [str] data directory.
        lab:        [str] laboratory from the meas comes from.
                    This will define the way to handle the data.
                    'kids-lab': Simon's KIDs lab.
                    'pete-lab': Peter Barry's lab.
        ----------
        """
        super().__init__()
        
        # Key arguments
        # ----------------------------------------------
        # project name
        project_name = kwargs.pop('project_name', None)
        # Working directory
        work_dir = kwargs.pop('work_dir', './')
        # cryostat
        cryostat = kwargs.pop('cryostat', 'aloysius')
        # meas_type
        meas_type = kwargs.pop('meas_type', None)
        # meas date
        meas_date = kwargs.pop('meas_date', None)
        # meas mode: auto/manual/prelim data
        meas_mode = kwargs.pop('meas_mode', None)
        # ----------------------------------------------

        # check frequency data format
        assert isinstance(directory, str), "Directory format not valid."

        # check if lab is valid
        assert lab in ['kids-lab', 'pete-lab'], "Laboratory not defined."

        # get features from foldername
        if directory[-1] == '/':
            directory = directory[:-1]

        foldername = directory.split('/')[-1]
        
        printc(f'Data type: {lab}', 'info')
        printc(f'Loading directory {directory}', 'info')

        if lab == 'kids-lab':
            # get properties from foldername
            foldername_features = self._get_kids_lab_filename_features(foldername)

            # assign variable values, prioritazing the user defined values.
            meas_params = []
            for i, item in enumerate([meas_type, meas_date, meas_mode]):
                param = item
                if item is None:
                    param = foldername_features[i]
                meas_params.append(param)
                printc(f'{param}', 'info')

            # assign meas main properties
            self.meas_type = meas_params[1].lower()
            self.meas_date = meas_params[0]
            self.meas_mode = meas_params[2]
            self.meas_comments = foldername_features[-1]

            self.data = self.load_kids_lab_data(directory, **kwargs)

        else:

            # initial definition
            if meas_type != 'dark':
                meas_type = 'dark'
                printc("So far, it is only possible to make dark testing in Pete's lab", 'info')

            # to get meas date
            if meas_date is None:
                date_directories = next(walk(directory), (None, None, []))[1]

                printc('Measurement date not defined', 'warn')

                if len(date_directories) > 1:
                    printc(f'Please select one the following folders:', 'info')
                    opts_valid = []
                    for i, date_diry in enumerate(date_directories):
                        printc(f'{i+1}. {date_diry}', 'info')
                        opts_valid.append(f'{i+1}')

                    date_diry_index = input()
                    while not date_diry_index in opts_valid:
                        printc('Option not valid, try again.', 'warn')
                        date_diry_index = input()

                    meas_date = date_directories[int(date_diry_index)-1]

                else:
                    meas_date = date_directories[0]

                printc(f'Measurement date assigned: {meas_date}', 'ok')

            self.meas_type = meas_type
            self.meas_date = meas_date
            self.meas_mode = meas_mode
            self.meas_comments = ""

            # define meas mode            
            if meas_mode is None:
                meas_mode = 'auto'

            self.data = self.load_pete_lab_data(os.path.join(directory, meas_date), **kwargs)

        printc('Done :)', 'ok')

        self.diry = directory
        self.lab = lab


    def load_pete_lab_data(self, directory, **kwargs):
        """
        Load Pete's lab data.
        Parameters
        ----------
        directory:      [str] path folder from where the data will be loaded.
        meas_date:      [str] measurement date.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # load only VNA data?
        only_vna = kwargs.pop('only_vna', True)
        # ----------------------------------------------

        # get the directories
        folders = next(walk(directory), (None, None, []))[1]

        assert not folders is None, "Directory doesn't exist, check again."
        assert len(folders) > 0, "Directory empty, folders not found."

        # get number of zeros for temperature prefix
        nzeros = self._temp_zeros
        # get temperature prefix
        temp_prefix = self._temp_prefix

        # get the data and store it in a dictionary
        data = {}
        
        # K I D
        for folder in folders:

            if folder == 'timestreams' and not only_vna:
                
                printc('L O A D I N G   T I M E S T R E A M S', 'info')

                time_folders = next(walk(os.path.join(directory, folder)), (None, None, []))[1]

                for time_folder in time_folders:

                    if time_folder.startswith('KID') and time_folder[3:].isnumeric():
                    
                        # initialise timestream object
                        if not 'hmd' in data:
                            data['hmd'] = {}
            
                        kid = 'K' + time_folder[3:].zfill(3)        # get kid name
                        data['hmd'][kid] = {}                       # initialise the timestream object per KID
            
                        # get sub folders
                        subfolders = next(walk(os.path.join(directory, folder, time_folder)), (None, None, []))[1]
                        
                        # T E M P E R A T U R E
                        for subfolder in tqdm(subfolders, desc='Loading detectors'):

                            temperature_units = subfolder[-2:]
                            temperature = temp_prefix + subfolder[:-2].zfill(nzeros)       # get temperature

                            # read the sub-subdiry
                            data['hmd'][kid][temperature] = {}
                            subsubfolders = next(walk(os.path.join(directory, folder, time_folder, subfolder)), (None, None, []))[1]
                            
                            # A T T E N U A T I O N
                            for subsubfolder in subsubfolders:
                                
                                attenuation = f'A{round(float(subsubfolder[:-2]), 1):.1f}'
                                data['hmd'][kid][temperature][attenuation] = {}

                                homodyne_data_path = os.path.join(directory, folder, time_folder, subfolder, subsubfolder)
                                data['hmd'][kid][temperature][attenuation]['path'] = homodyne_data_path

                                # get all the files
                                files = os.listdir(homodyne_data_path)
                                nmeas = [0, 0]
                                nsweep = 0

                                printc(f'{kid}-{temperature}-{attenuation}', 'title3')
                                files.sort()

                                for file in files:

                                    # read data
                                    sample_data = np.load(os.path.join(homodyne_data_path, file), allow_pickle=True)

                                    # get general header
                                    gral_hdr = sample_data['state_variables'].item().copy()

                                    if "on" in file.lower():
                                        noise_mode = "on"
                                        nsample = nmeas[0]

                                    elif "off" in file.lower():
                                        noise_mode = "off"
                                        nsample = nmeas[1]
                                    
                                    if "sweep" in file:         # sweep data
                                        
                                        # get sweep data
                                        frequency = sample_data['frequency_data']

                                        s21_data = sample_data['complex_data'][0]
                                        I, Q = s21_data.real, s21_data.imag

                                        hdr = {}        # <--- N E E D S   T O   B E   D E F I N E D
                                        hdr['DATE'] = sample_data['date']
                                        hdr['SWEEPTYP'] = "noise"
                                        hdr['TIME'] = ""
                                        hdr['DUT'] = sample_data['dut']
                                        hdr['ATT_RT'] = 0       # <--- 0 BY NOW, BUT NEED TO BE CHANGED
                                        hdr['BLACKBOD'] = 0
                                        hdr['SAMPLETE'] = gral_hdr['eagle']['parameters']['temperature']['value'][6]

                                        if 'on' in data['hmd'][kid][temperature][attenuation]:
                                            hdr['F0'] = data['hmd'][kid][temperature][attenuation]['on'][0].hdr['F0']

                                        # create vna data objects
                                        sweep_full_object = sweep(frequency, I, Q, hdr.copy())

                                        # create sweep object per each detector
                                        #hdr['F0'] = hdr['F0FOUND']
                                        sweep_object = sweep(frequency, I, Q, hdr.copy())

                                        if not "sweep" in data['hmd'][kid][temperature][attenuation]:
                                            data['hmd'][kid][temperature][attenuation]["sweep"] = {}

                                        data['hmd'][kid][temperature][attenuation]["sweep"][nsweep] = sweep_object
                                        nsweep += 1

                                    elif "timestream" in file:  # noise data

                                        if not noise_mode in data['hmd'][kid][temperature][attenuation]:
                                            data['hmd'][kid][temperature][attenuation][noise_mode] = {}

                                        # read the noise data
                                        signal = sample_data['iq_timestream']
                                        nsamples = signal.shape[0]

                                        I, Q = [], []                                        
                                        for sample in range(nsamples):
                                            I.append(signal[sample].real)
                                            Q.append(signal[sample].imag)

                                        noise_hdr = {}        # <--- N E E D S   T O   B E   D E F I N E D
                                        noise_hdr['NOISETYP'] = noise_mode
                                        noise_hdr['DATE'] = sample_data['date']
                                        noise_hdr['TIME'] = ""
                                        noise_hdr['DUT'] = sample_data['dut']
                                        noise_hdr['ATT_RT'] = 0       # <--- 0 BY NOW, BUT NEED TO BE CHANGED
                                        noise_hdr['BLACKBOD'] = 0
                                        noise_hdr['SAMPLETE'] = gral_hdr['eagle']['parameters']['temperature']['value'][6]
                                        noise_hdr['F0'] = gral_hdr['smf']['parameters']['frequency']['value']
                                        noise_hdr['SYNTHPOW'] = gral_hdr['smf']['parameters']['power']['value']
                                        fs = gral_hdr['daq']['parameters']['sampling_rate']['value']
                                        noise_hdr['SAMPLERA'] = fs
                                        noise_hdr['SAMPLELE'] = len(I[0])/fs

                                        # create noise object per each detector
                                        noise_object = noise(I, Q, noise_hdr.copy())

                                        data['hmd'][kid][temperature][attenuation][noise_mode][nsample] = noise_object
                                        
                                        if noise_mode == "on":
                                            nmeas[0] += 1
                                        
                                        elif noise_mode == "off":
                                            nmeas[1] += 1

            elif folder == 'vna_sweeps':
                
                printc('L O A D I N G   S W E E P S', 'info')

                # for vna sweeps
                data['vna'] = {}

                # for continuous full sweeps
                data['vna']['full'] = {}

                # get all the files
                path_files_list = os.listdir(os.path.join(directory, folder))
                path_files_list = [f for f in path_files_list if os.path.isfile( os.path.join(directory, folder, f) ) ]

                print(path_files_list)

                for file in path_files_list:

                    if file == 'full_tones_list.npz':
                        printc("Read tone list, I don't know how yet", 'warn')

                    else:
                        
                        # read data
                        sample_data = np.load(os.path.join(directory, folder, file), allow_pickle=True)
                        # get measurement properties
                        sweep_mode, temperature, attenuation, nsample = self._get_pete_lab_filename_features(file)

                        # check if data is any kind of sweep: continuous or segmented.          
                        if sweep_mode.lower() in ['segmented', 'continuous']:

                            temperature = temp_prefix + temperature.zfill(nzeros)       # get temperature label
                            attenuation = f'A{attenuation}' 

                            # get sweep data
                            frequency = sample_data['frequency_data'][0]

                            s21_data = sample_data['complex_data'][0]
                            I, Q = s21_data.real, s21_data.imag

                            # get general header
                            gral_hdr = sample_data['state_variables'].item()
                            data['vna']['header'] = gral_hdr

                            hdr = {}        # <--- N E E D S   T O   B E   D E F I N E D
                            hdr['SWEEPTYP'] = sweep_mode
                            hdr['DATE'] = sample_data['date']
                            hdr['TIME'] = ""
                            hdr['DUT'] = sample_data['dut']
                            hdr['ATT_RT'] = 0       # <--- 0 BY NOW, BUT NEED TO BE CHANGED
                            hdr['BLACKBOD'] = 0
                            hdr['SAMPLETE'] = gral_hdr['eagle']['parameters']['temperature']['value'][6]

                            # segmented data
                            if sweep_mode == 'segmented':

                                nkids = len(frequency)      # get number of detectors available

                                for kid in range(nkids):
                                    # kid name                                
                                    kid_name = 'K'+str(kid).zfill(3)   

                                    if not kid_name in data['vna']:
                                        data['vna'][kid_name] = {}

                                    if not temperature in data['vna'][kid_name]:
                                        data['vna'][kid_name][temperature] = {}

                                    if not attenuation in data['vna'][kid_name][temperature]:
                                        data['vna'][kid_name][temperature][attenuation] = {}

                                    # create sweep object
                                    sweep_seg_object = sweep(frequency[kid], I[kid], Q[kid], hdr.copy())

                                    data['vna'][kid_name][temperature][attenuation][nsample] = sweep_seg_object

                            # continuous sweep
                            if sweep_mode == 'continuous':

                                # create vna data objects
                                sweep_full_object = sweep(frequency, I, Q, hdr.copy())

                                if not temperature in data['vna']['full']:
                                    data['vna']['full'][temperature] = {}
                                
                                if not attenuation in data['vna']['full'][temperature]:
                                    data['vna']['full'][temperature][attenuation] = {}

                                data['vna']['full'][temperature][attenuation][nsample] = sweep_full_object


        return data


    def load_kids_lab_data(self, directory, trim_edge=0.25, **kwargs):
        """
        Load timestream + vna sweeps data from KIDs lab.
        Parameters
        ----------
        directory:          [str] path folder from where the data will be loaded.
        trim_edges(opt):    [float] trim edges.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # load only VNA data?
        only_vna = kwargs.pop('only_vna', True)
        # frequency threshold to distangle segmented sweeps
        freq_segmented_thresh = kwargs.pop('freq_segmented_thresh', 1e4)
        # expected resonators for the segmented sweep
        expected_kids = kwargs.pop('expected_kids', None)
        # ----------------------------------------------

        # get the directories
        folders = next(walk(directory), (None, None, []))[1]

        assert not folders is None, "Directory doesn't exist, check again."
        assert len(folders) > 0, "Directory empty, folders not found."

        # get number of zeros for temperature prefix
        nzeros = self._temp_zeros
        # get temperature prefix
        temp_prefix = self._temp_prefix

        # get the data and store it in a dictionary
        data = {}
        
        # K I D
        for folder in folders:

            if folder.startswith('KID_K') and folder[5:].isnumeric():
            
                # read homodyne data
                if not only_vna:

                    # initialise timestream object
                    if not 'hmd' in data:
                        data['hmd'] = {}
        
                    kid = folder[4:]        # get kid name
                    data['hmd'][kid] = {}    # initialise the timestream object per KID
        
                    # get sub folders
                    subfolders= next(walk(os.path.join(directory, folder)), (None, None, []))[1]
                    
                    # T E M P E R A T U R E
                    for subfolder in tqdm(subfolders, desc='Loading detectors'):

                        temperature_units = subfolder.split('_')[3]
                        temperature = temp_prefix + subfolder.split('_')[2].zfill(nzeros)       # get temperature

                        # read the sub-subdiry
                        data['hmd'][kid][temperature] = {}
                        subsubfolders = next(walk(os.path.join(directory, folder, subfolder)), (None, None, []))[1]
                        
                        # A T T E N U A T I O N
                        for subsubfolder in subsubfolders:
                            
                            attenuation = f'A{round(float(subsubfolder[16:-2]), 1):.1f}'
                            data['hmd'][kid][temperature][attenuation] = {}

                            homodyne_data_path = os.path.join(directory, folder, subfolder, subsubfolder)
                            data['hmd'][kid][temperature][attenuation]['path'] = homodyne_data_path

                            # get all the files in directory
                            sweep_path, sweep_hr_path, noise_on_path, noise_off_path = get_homodyne_files(homodyne_data_path)

                            # loading sweeps
                            # -------------
                            sweep_modes = ['sweep', 'sweep_hr']
                            for s, sweep_paths in enumerate([sweep_path, sweep_hr_path]):
                                
                                sweep_mode = sweep_modes[s]
                                
                                for sweep_path_file in sweep_paths:

                                    if not sweep_mode in data['hmd'][kid][temperature][attenuation]:
                                        data['hmd'][kid][temperature][attenuation][sweep_mode] = {}

                                    filename = os.path.join(homodyne_data_path, sweep_path_file[0])
                                    nsample = sweep_path_file[1]

                                    # read the sweep data
                                    sweep_hdr, s21_sweep = self.load_fits(filename, index=1)
                                    sweep_hdr['SWEEPTYP'] = "noise"

                                    freq = s21_sweep['Frequency']
                                    I, Q = s21_sweep['RawI'], s21_sweep['RawQ']

                                    # create sweep object per each detector
                                    sweep_hdr['F0'] = sweep_hdr['F0FOUND']
                                    sweep_object = sweep(freq, I, Q, sweep_hdr.copy())

                                    data['hmd'][kid][temperature][attenuation][sweep_mode][nsample] = sweep_object

                            # loading noise data
                            # -------------
                            noise_modes = ['on', 'off']
                            for s, noise_paths in enumerate([noise_on_path, noise_off_path]):
                                
                                noise_mode = noise_modes[s]
                                nsample = 0
                                for noise_path_file in noise_paths:

                                    if not noise_mode in data['hmd'][kid][temperature][attenuation]:
                                        data['hmd'][kid][temperature][attenuation][noise_mode] = {}

                                    filename = os.path.join(homodyne_data_path, noise_path_file[0])
                                    #nsample = noise_path_file[-1]

                                    # read the noise data
                                    noise_hdr, noise_data = self.load_fits(filename, index=1)

                                    noise_hdr['NOISETYP'] = noise_mode

                                    I = [noise_data.field(2*i) for i in range(int(len(noise_data[0])/2))]
                                    Q = [noise_data.field(2*i+1) for i in range(int(len(noise_data[0])/2))]

                                    # create noise object per each detector
                                    noise_object = noise(I, Q, noise_hdr.copy())

                                    data['hmd'][kid][temperature][attenuation][noise_mode][nsample] = noise_object

                                    # this is a big change of how I understand the multiple fs data, former high/low resolution noise data
                                    # because it is a bit ambiguous, what if I have more than 2, would I need to define intermedians?
                                    # the solution consist in define them as multiple samples.
                                    nsample += 1

            # read VNA data
            elif folder == 'VNA_Sweeps':

                # for vna sweeps
                data['vna'] = {}

                # continuous full vna
                data['vna']['full'] = {}
                
                # get vna files
                vna_files = next(walk(os.path.join(directory, folder)), (None, None, []))[2]

                for vna_file in tqdm(vna_files, desc='Loading VNA files'):

                    # get data name
                    if vna_file.startswith('S21_'):

                        sweep_mode, temperature, temperature_units, attenuation, nsample = self._get_vna_features_by_filename(vna_file)

                        temperature = temp_prefix + temperature         # temperature field name
                        attenuation = f'A{attenuation}'                 # attenuation field name

                        # read vna sweep
                        sweep_full_path = os.path.join(directory, folder, vna_file)
                        gral_hdr, s21_data = self.load_fits(sweep_full_path, index=1)

                        freq = s21_data['Freq']
                        I, Q = s21_data['ReS21'], s21_data['ImS21']

                        # get general header
                        data['vna']['header'] = gral_hdr

                        # local header per kid/continuous sweep
                        hdr = {}        
                        hdr['SWEEPTYP'] = sweep_mode
                        hdr['DATE'] = gral_hdr['DATE']
                        hdr['TIME'] = gral_hdr['TIME']
                        hdr['DUT'] = gral_hdr['DUT']
                        hdr['ATT_RT'] = gral_hdr['ATT_RT']
                        hdr['BLACKBOD'] = gral_hdr['BLACKBOD']
                        hdr['SAMPLETE'] = gral_hdr['SAMPLETE']

                        # read continuous sweeps
                        if sweep_mode == 'con':

                            # create vna data objects
                            sweep_full_object = sweep(freq, I, Q, hdr.copy())

                            if not temperature in data['vna']['full']:
                                data['vna']['full'][temperature] = {}
                            
                            if not attenuation in data['vna']['full'][temperature]:
                                data['vna']['full'][temperature][attenuation] = {}
                            
                            data['vna']['full'][temperature][attenuation][nsample] = sweep_full_object
                            #data['vna']['full'][temperature][attenuation][nsample] = sweep_full_path

                        # read segmented data
                        elif sweep_mode == 'seg':

                            # get number of kids in segmented sweep
                            n_kids = self.find_kids_segmented_vna(freq, freq_thresh=freq_segmented_thresh, expected=expected_kids)

                            # split data per KID
                            for kid in range(len(n_kids)-1):
                                
                                # get detector range
                                a_idx = n_kids[kid]+1
                                b_idx = n_kids[kid+1]

                                # get individual detector data
                                freq_kid = freq[a_idx:b_idx]
                                I_kid, Q_kid = I[a_idx:b_idx], Q[a_idx:b_idx]
                                data_len = len(freq_kid)

                                # trim data <--- C H E C K   I F   N E E D   I T
                                trim_section = int(trim_edge*data_len)
                                freq_kid = freq_kid[trim_section:data_len-trim_section]
                                I_kid = I_kid[trim_section:data_len-trim_section]
                                Q_kid = Q_kid[trim_section:data_len-trim_section]
                                
                                # kid name                                
                                kid_name = 'K'+str(kid).zfill(3)       

                                if not kid_name in data['vna']:
                                    data['vna'][kid_name] = {}

                                if not temperature in data['vna'][kid_name]:
                                    data['vna'][kid_name][temperature] = {}

                                if not attenuation in data['vna'][kid_name][temperature]:
                                    data['vna'][kid_name][temperature][attenuation] = {}

                                # create sweep object per each detector
                                sweep_per_kid_object = sweep(freq_kid, I_kid, Q_kid, hdr.copy())

                                data['vna'][kid_name][temperature][attenuation][nsample] = sweep_per_kid_object
                                
        return data


    def find_kids_segmented_vna(self, freq, freq_thresh=1e4, expected=None):
        """
        Find the KIDs in the segmented VNA.
        Parameters
        ----------
        freq:               [array] frequency array [Hz].
        freq_thresh[opt]:   [float] KID detection threshold.
        expected[opt]:      [float] expected resonators.
                            If None, use the threshold to identify them.
        ----------
        """

        if not expected is None:

            segs = np.reshape(np.arange(len(freq)), (expected, int(len(freq)/expected)))
            n_kids = np.zeros(expected, dtype=int)

            for i in range(expected):
                n_kids[i] = segs[i][0]

            n_kids = n_kids[1:]

        else:
            # number of KIDs
            n_kids = np.where( np.abs(np.diff(freq)) > freq_thresh )[0]

        n_kids = np.concatenate(([0], n_kids, [-1]))

        return n_kids
    

    def _get_kids_lab_filename_features(self, foldername):
        """
        Get the date, analysis type and sample from the folder name.
        Parameters
        ----------
        foldername:     [str] foldername from where data is extracted.
        ----------
        """

        items = foldername.split('_')

        features = []
        comment_field = ""
        for i, item in enumerate(items):
            if i > 3:
                comment_field += f'{item} '
            else:
                features.append(item)

        features.append(comment_field[:-1])

        return features


    def _get_vna_features_by_filename(self, filename, **kwargs):
        """
        Extract the temperature and attenuation from the VNA name.
        Parameters
        ----------
        filename:       [str] filename from where the temp and 
                        attenuation are extracted.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Temperature subfix
        temp_subfix = kwargs.pop('temp_subfix', 'temp')
        # Attenuation subfix
        atten_subfix = kwargs.pop('atten_subfix', 'att')
        # ----------------------------------------------

        assert isinstance(filename, str), "Filename has to be a string variable."

        filename = filename.lower()     # force to lowercase

        if 'segmented' in filename:
            vna_type = 'seg'
        elif 'continuous' in filename:
            vna_type = 'con'
        else:
            vna_type = 'unkwnown'

        try:
            # get the temperature
            start_idx_temp = filename.index(temp_subfix)
            end_idx_temp = filename[start_idx_temp:].index('_')+start_idx_temp

            temp = filename[start_idx_temp+len(temp_subfix):end_idx_temp]
            temp_units = filename.split('_')[3]

            # get the attenuation
            start_idx_atten = filename.index(atten_subfix)
            end_idx_atten = filename[start_idx_atten:].index('_')+start_idx_atten
            
            atten = filename[start_idx_atten+len(atten_subfix):end_idx_atten]

            # sample number
            sample_number = filename.split('_')[-1].split('.')[0]

            if sample_number.isnumeric():
                number_sample = int(sample_number)
            else:
                number_sample = 0

            return vna_type, temp, temp_units, atten, number_sample

        except ValueError as err:

            printc(f"{err}\nFilename format not expected", 'fail')
            return 0

    @property
    def _temp_zeros(self):
        """ Get number of zeros for temperature prefixes. """

        # blackbody analysis by default
        nzeros = 3
        # if dase temperature test
        if self.meas_type == 'dark':
            nzeros = 4

        return nzeros

    @property
    def _temp_prefix(self):
        """ Get temperature prefix. """

        # blackbody analysis by default
        temp_prefix = 'B'
        # if dase temperature test
        if self.meas_type == 'dark':
            temp_prefix = 'D'

        return temp_prefix


    def _create_diry(self, diry):
        """
        Create a directory.
        Parameters
        ----------
        diry:       [str] path and name of the new folder
        ----------
        """

        try:
            
            folders = diry.split('/')

            accum_path = ""
            if diry[0] == "/":
                accum_path = "/"

            for folder in folders:
                # get accumulate path
                accum_path = os.path.join(accum_path, folder)
                # chek if folder exist
                if not os.path.exists(accum_path):
                    # create folder
                    os.system('mkdir '+accum_path)

        except Exception as e:
            printc('Directory not created. '+str(e), 'warn')


    def _get_pete_lab_filename_features(self, filename):
        """ Get features from Pete's lab filenames. """

        # features container: mode (continuous/segmented), temperature, attenuation, sample
        features = [None, 0, 0, 0]

        for i, item in enumerate(filename.split('_')):

            if item.endswith('mK'):
                features[1] = item[:-2]

            elif item.endswith('K'):
                features[1] = item[:-1]

            elif item.endswith('dB'):
                features[2] = item[:-2]
            
            elif item.endswith('.npz'):
                sample = item[:-4]
                if sample.isnumeric():
                    features[3] = int(sample)

            elif i == 0:
                features[0] = item

        return features
