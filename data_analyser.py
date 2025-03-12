# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID-analysis. Data Analyser
# data_analyser.py
# Main class to analyse KIDs data: noise and s21 sweep.
#
# Marcial Becerril, @ 14 February 2025
# Latest Revision: 14 Feb 2025, 16:19 UTC
#
# TODO list:
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


import os
import numpy as np

from multiprocessing import Process, Manager
fitRes = Manager().dict()

from misc.funcs import *
from physics.funcs import *
from misc.fits_files_handler import *
from data_processing import *

from s21_meas import *
from noise_meas import *
from homodyne_functions import *
from data_manager import *
from fit_single_resonator import *
from fit_single_psd import *
from overdriven_figures import *

from matplotlib.pyplot import *

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.family'] = 'serif'



# M A N A G E R   C L A S S
# -----------------------------------
class KidAnalyser(DataManager):
    """ Data analysis class. """

    def __init__(self, directory, **kwargs):
        """
        Parameters
        ----------
        directory:  [str] data directory.
        ----------
        """
        super().__init__(directory, **kwargs)
        
        # Key arguments
        # ----------------------------------------------
        # project name
        project_name = kwargs.pop('project_name', "M1-test")
        # working directory
        work_dir = kwargs.pop('work_dir', './')
        # ----------------------------------------------

        self.project_name = project_name        # get project name
        self.work_dir = work_dir                # get working directory

        # overdriven attenuation for s21 sweeps
        self.sweep_overdriven = []
        self.load_opt_attens()

        # load fit resoantors results
        try:
            self.load_fit_resonators()

        except Exception as e:
            printc(f'It was not possible to load fit results.\n{e}', 'fail')

        # create some basic folders
        #if not os.path.isfile(f'{self.work_dir}{self.project_name}/misc/'):
        self._create_diry(f'{self.work_dir}{self.project_name}/misc/')

        # extra attenuation
        self.add_input_atten = 0

        # responsivity calculation params
        self.res_params = {}

        # noise params
        self.psd_params = {}


    def get_dx_dP(self, kid, attenuation=None, temperatures="all", sample=0, res_meas='f0', power_meas="temp",
                  res_method="fit", power_method="bw", meas='vna', phase_ref_temp='min', material='Al', 
                  volume=1, modes=1, **kwargs):
        """
        Get dx vs dP (power)
        ----------------
        kid:            [int/str] detector id.
        temperature:    [int/str] temperature.
        attenuation:    [int/str] attenuation.
        sample:         [int/str] samples. If 'all', select all the samples.
        res_meas:       [str] responsivity measurable variable: resonance frequency ['f0'], phase ['phase'], or quality factor ['Qr']
        ----------------
        """
        # Key arguments
        # ----------------------------------------------
        # spectrum bandwidth (assuming top hat) [Hz]
        spectrum_bw = kwargs.pop('spectrum_bw', 30e9)
        # spectrum from FTS, file path
        fts_path = kwargs.pop('fts_path', None)
        # fts frequency range in GHz
        fts_freq_range = kwargs.pop('fts_freq_range', [0, 350])
        # smooth abs(s21) to get f0 from minimum
        s21_smooth_params = kwargs.pop('s21_smooth_params', [31, 3])
        # smooth i,q to get phase shift
        circle_smooth_params = kwargs.pop('circle_smooth_params', None)
        # ----------------------------------------------

        assert meas in ['vna', 'hmd'], "Measurement system is not defined.\nOnly 'vna' or 'hmd' are valid."
        
        assert res_method in ['fit', 'raw_min', 'sweep'], "Method to extract response not valid."
        assert power_method in ['bw', 'fts'], "Power method not valid.\nChoose one of the followings: 'bw' (bandwidth) or 'fts'."

        assert res_meas in ['f0', 'phase', 'Qi'], "Responsivity variable not valid.\nChoose one of the followings: 'f0', 'phase', or 'Qi'." 
        assert power_meas in ['temp', 'power', 'Nqp'], "Power variable not valid.\nChoose one of the followings: 'temp', 'power', or 'Nqp'." 

        # parse key format
        k = self._parse_single_item(kid, prefix='K', nzeros=3)                # get kid

        if attenuation == None:
            attenuation = self.sweep_overdriven[int(k[1:])]

        a = self._parse_single_item(attenuation, prefix='A', nzeros=3)        # get attenuation

        # get temperatures
        if temperatures == "all":
            selected_temps = np.sort(list(self.data[meas][k].keys()))
        else:
            selected_temps = self._parse_items(temperatures, prefix=self._temp_prefix, nzeros=self._temp_zeros)

        if phase_ref_temp == 'min':
            temp_ref = selected_temps[0]

        # save status
        self.res_params[k] = {
            "x": {
                "var": res_meas,
                "method": res_method,
                "phase_ref_temp": temp_ref,
                "s21_smooth_params": s21_smooth_params,
                "circle_smooth_params": circle_smooth_params,
            },
            "power": {
                "type": power_meas,
                "method": power_method,
                "modes": modes,
                "bw": spectrum_bw,
                "fts": {
                    "path": fts_path,
                    "freq_range": fts_freq_range,
                },
            },
            "detector_properties":{
                "material": material,
                "volume": volume,
            },
            "meas": meas,
        }

        x0s, meas_temps, powers = [], [], []

        if meas == 'vna':

            f0_ref = self.data[meas][k][temp_ref][a][sample].hdr['F0']

            xc = self.data[meas][k][temp_ref][a][sample].xc.copy()
            yc = self.data[meas][k][temp_ref][a][sample].yc.copy()
            theta = self.data[meas][k][temp_ref][a][sample].theta.copy()

        else:

            f0_ref = self.data[meas][k][temp_ref][a]['sweep'][sample].hdr['F0']

            xc = self.data[meas][k][temp_ref][a]['sweep'][sample].xc.copy()
            yc = self.data[meas][k][temp_ref][a]['sweep'][sample].yc.copy()
            theta = self.data[meas][k][temp_ref][a]['sweep'][sample].theta.copy()

        rot_params = [xc, yc, theta]

        fig_res, axs_res = subplots(1, 2, num=f'res-{k}-{a}', figsize=(18, 10))
        fig_res.subplots_adjust(top=0.97, bottom=0.1, left=0.075, right=0.98)

        # Check selected temperatures
        for temperature in selected_temps:

            # extract the object
            if meas == 'vna':
                meas_obj = self.data[meas][k][temperature][a][sample]
            else:
                meas_obj = self.data[meas][k][temperature][a]['sweep'][sample]

            # if dark measurement get bath temperature
            if self._temp_prefix == "D":
                meas_temp = meas_obj.hdr['SAMPLETE']
                temp_units = 'mK'

            # if optical measurement get blackbody temp
            else:
                meas_temp = meas_obj.hdr['BLACKBOD']
                temp_units = 'K'

            axs_res[0].plot(1e-6*meas_obj.frequency, 20*np.log10(np.abs(meas_obj.s21_nr)) )

            # get response variable
            # ----------------------------------------
            # f0: resonance frequency
            if res_meas == 'f0':
                
                x0, s21_sm = self.get_f0_from_sweep(meas_obj, f0_method=res_method, smooth_params=s21_smooth_params)

                # plot
                axs_res[1].plot(meas_obj.s21_nr.real, meas_obj.s21_nr.imag, label=f'{meas_temp:.1f} {temp_units}, {a[1:]} dB')

                if res_method == 'fit' and meas_obj.fit != {}:
                    axs_res[0].plot(1e-6*meas_obj.frequency, 20*np.log10(np.abs(meas_obj.fit['fit_sweep'])), 'k', lw=1.25 )
                    axs_res[1].plot(meas_obj.fit['fit_sweep'].real, meas_obj.fit['fit_sweep'].imag, 'k', lw=1.25)
                
                elif res_method == "raw_min":
                    axs_res[0].plot(1e-6*meas_obj.frequency, 20*np.log10(s21_sm), 'k')

            # phase
            elif res_meas == 'phase':

                x0, I0, Q0, I_derot, Q_derot, I_sm, Q_sm = self.get_phase_shift(meas_obj, f0_ref=f0_ref, rot_params=rot_params,
                                                               smooth_params=circle_smooth_params)

                axs_res[1].plot(I_derot, Q_derot, label=f'{meas_temp:.1f} {temp_units}, {a[1:]} dB')

                if circle_smooth_params != None:
                    axs_res[1].plot(I_sm, Q_sm, 'k', lw=1)
                
                axs_res[1].plot(I0, Q0, 'rs')

            # quality factor
            elif res_meas == 'Qi':
                
                axs_res[1].plot(meas_obj.s21_nr.real, meas_obj.s21_nr.imag, label=f'{meas_temp:.1f} {temp_units}, {a[1:]} dB')

                if meas_obj.fit != {}:
                    x0 = meas_obj.fit['Qi']

                else:
                    x0 = None

            # get power variable
            # ----------------------------------------
            # temperature
            if power_meas == 'temp':

                power = meas_temp

                printc(f'Measured BB temperature {meas_temp:.1f} K', 'ok')

            elif power_meas == 'power':

                if power_method == 'bw':

                    power = bb2pwr(meas_temp, spectrum_bw)

                elif power_method == 'fts':

                    # read fts data
                    assert os.path.isfile(fts_path), f"FTS file {fts_path} not found."

                    freq_fts, tx_fts = load_fts(fts_path, freq_range=fts_freq_range)

                    power = get_power_from_FTS(freq_fts, tx_fts, meas_temp, modes=1)
            
                printc(f'Total power at {meas_temp:.1f} K: {1e12*power:.2f} [pW]', 'ok')

            elif power_meas == 'Nqp':

                Tc = Tcs[material]
                N0 = N0s[material]

                # get energy gap from Tc
                Delta = get_Delta(Tc)
                # get density of quasiparticles
                nqp = get_nqp(N0, meas_temp, Delta)
                # get number of quasiparticles
                Nqp = nqp * volume

                power = Nqp

                printc(f'Nqp at {1e3*meas_temp:.1f} mK: {power:,.1f}', 'ok')

            # collect results
            # ----------------------------------------
            if x0 != None:

                x0s.append(x0)
                powers.append(power)
                meas_temps.append(meas_temp)

                printc(f'{k}-{temperature}-{a}-{sample}', 'ok')

            axs_res[0].set_xlabel(f'Frequency [MHz]')
            axs_res[0].set_ylabel(f'S21 [dB]')
            axs_res[0].grid()

            axs_res[1].set_xlabel(f'I [V]')
            axs_res[1].set_ylabel(f'Q [V]')
            axs_res[1].axhline(0, color="k", lw="1.5")
            axs_res[1].axvline(0, color="k", lw="1.5")
            axs_res[1].grid()

            axs_res[1].legend(fontsize=11)

        # save responsivity parameters
        self._create_diry(f'{self.work_dir}{self.project_name}/misc/')
        np.save(f'{self.work_dir}{self.project_name}/misc/responsivity_params', self.res_params)

        # get gradient
        dx_dP = np.gradient(x0s, powers)

        return dx_dP, np.array(x0s), np.array(powers), np.array(meas_temps)


    def get_phase_shift(self, meas_sweep, f0_ref, rot_params, smooth_params=None):
        """
        Get the phase shift in radians.
        Parameters
        ----------
        meas_sweep:     [object] sweep object.
        f0_ref:         [float] reference resonance frequency.
        ----------
        """

        xc, yc, theta = rot_params
            
        I_derot, Q_derot = derot_circle(meas_sweep.s21.real, meas_sweep.s21.imag, xc, yc, theta)

        # get I, Q at f0_ref
        frequency = meas_sweep.frequency
        f0_idx = np.argmin( np.abs(frequency - f0_ref) )

        if smooth_params == None:
            I_sm = I_derot
            Q_sm = Q_derot
        
        else:
            I_sm = savgol_filter( I_derot, *smooth_params )
            Q_sm = savgol_filter( Q_derot, *smooth_params )

        I0 = I_sm[f0_idx]
        Q0 = Q_sm[f0_idx]

        phase_f0 = np.arctan2(Q0, I0)

        if phase_f0 > np.pi/4:
            phase_f0 = phase_f0 - 2*np.pi

        return phase_f0, I0, Q0, I_derot, Q_derot, I_sm, Q_sm


    def get_f0_from_sweep(self, meas, f0_method='raw_min', smooth_params=[31, 3]):
        """
        Get f0 from a sweep object.
        Parameters
        ----------
        meas:       [object] sweep object.
        smooth:     [list] smooth savgol paramaters for sweep.
        ----------
        """

        # smooth sweep array
        s21_sm = np.array([])

        if f0_method == "fit" and meas.fit != {}:
            
            f0 = meas.fit['fr']

        elif f0_method == "sweep":

            f0 = meas.hdr['F0']

        elif f0_method == "raw_min":
            
            s21 = np.abs(meas.s21_nr)
            s21_sm = savgol_filter(s21, *smooth_params)

            f0 = meas.frequency[np.argmin(s21_sm)]

        else:

            f0 = None

        return f0, s21_sm


    def load_opt_attens(self, path=None):
        """ Load optimal attenuations. """

        if path == None:
            over_atten_path = f'{self.work_dir}{self.project_name}/misc/over_attens.npy'

        else:
            over_atten_path = path

        if os.path.isfile(over_atten_path):
            self.sweep_overdriven = np.load(over_atten_path, allow_pickle=True)
            printc(f"Optimal attenuations loaded.", "ok")

        else:
            printc(f"Optimal attenuations file not found.", "fail")


    def save_opt_attens(self, filename=None):
        """ Save optimal attenuations. """

        if filename == None:
            self._create_diry(f'{self.work_dir}{self.project_name}/misc/')
            filename = f"{self.work_dir}{self.project_name}/misc/over_attens.npy"

        try:
            np.save(filename, self.sweep_overdriven)
            printc("Optimal attenuations saved.", "ok")
        
        except:
            printc("Optimal attenuations not saved", "fail")


    def fit_resonators(self, kids="all", temps="all", attens="all", samples="all",
                       meas='vna', overwrite=False, n=3.5, **kwargs):
        """
        Fit the resonators in frequency domain.
        Define a set of data to fit. If none is defined, it will
        fit all the detectors.
        Parameters
        ----------
        kids:           [list/str] KID ID. If 'all' it will take all the resonators.
        temps:          [list/str] temperature. 
                        If 'all' it will take all the temperatures, whether base
                        temperature or Blackbody temperature.
        attens:         [list/str] attenuation. If 'all' it will select all the attenuations.
        samples:        [list/str] samples. If 'all', select all the samples.
        meas:           [str] measurements type: 'vna' or 'hmd' (homodyne).
        overwrite:      [bool] If the defined measurements are not fit, finish the job.
                        It won't repeat the fit, where is hs been done, unless it is False.
        n(opt):         [float] number of linewidths to select data.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Cable delay
        tau = kwargs.pop('tau', 50e-9)
        # Plot results
        plot_res = kwargs.pop('plot_res', True)
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        assert meas in ['vna', 'hmd'], "Measurement system is not defined.\nOnly 'vna' or 'hmd' are valid." 
       
        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        for kid in selected_kids:

            printc(f'{kid}', 'info')
            
            # get temperatures
            if temps == "all":
                selected_temps = self.data[meas][kid].keys()
            else:
                selected_temps = self._parse_items(temps, prefix=self._temp_prefix, nzeros=self._temp_zeros) 
            
            fit_jobs = []       # fitting jobs list

            for temp in selected_temps:

                printc(f'\t-->{temp}', 'info')

                # get attenuations
                if attens == "all":
                    selected_attens = self.data[meas][kid][temp].keys()
                else:
                    selected_attens = self._parse_items(attens, prefix='A', nzeros=3)
                
                for atten in selected_attens:
                
                    if float(atten[1:]) >= self.sweep_overdriven[int(kid[1:])]:

                        printc(f'\t\t--->{atten}', 'info')

                        if samples == "all":
                            if meas == 'vna':
                                selected_samples = self.data[meas][kid][temp][atten].keys()
                            else:
                                selected_samples = self.data[meas][kid][temp][atten]['sweep'].keys()
                        else:
                            selected_samples = samples.copy()

                        for sample in selected_samples:

                            # check measurement is available
                            available_sample = self._check_availability(kid, temp, atten, sample=sample, meas_type=meas)

                            if available_sample:

                                # define measurement
                                if meas == 'vna':
                                    meas_obj = self.data['vna'][kid][temp][atten][sample]
                                
                                else:
                                    meas_obj = self.data['hmd'][kid][temp][atten]['sweep'][sample]

                                # check if the sample is fitable
                                flag_do_fit = False
                                
                                if overwrite:
                                    flag_do_fit = True

                                elif not overwrite and meas_obj.fit == {}:
                                    flag_do_fit = True

                                # do the fit
                                if flag_do_fit:

                                    try:
                                        # get frequency and s21
                                        frequency = meas_obj.frequency
                                        s21 = meas_obj.s21

                                        # run the fit in parallel jobs
                                        p = Process(target=self._fit_res_join, args=(kid, temp, atten, frequency, s21, n, tau ))
                                        fit_jobs.append(p)
                                        p.start()

                                    except KeyError as key_err:
                                        printc(f'Data not found.\nKey {key_err} is not defined.', 'fail')

                                    except Exception as e:
                                        printc(f'Fit is taking too much time, operation canceled.\n{e}', 'fail')

                                else:
                                    printc(f'{kid}-{temp}-{atten}-{sample} done', 'ok')

                            else:
                                printc(f"Data sample {kid}-{temp}-{atten}-{sample} not available.", "warn")

            # collect jobs
            for proc in fit_jobs:
                proc.join()
            
            # get fit results
            for sweep_sample in fitRes.keys():
                
                # get key measurement
                keys_meas = sweep_sample.split(',')
                
                kid     = keys_meas[0]
                temp    = keys_meas[1]
                atten   = keys_meas[2]
            
                # define measurement
                if meas == 'vna':
                    self.data[meas][kid][temp][atten][sample].fit = fitRes[sweep_sample].copy()
                    meas_obj = self.data[meas][kid][temp][atten][sample]
                
                elif meas == 'hmd':
                    self.data[meas][kid][temp][atten]['sweep'][sample].fit = fitRes[sweep_sample].copy()
                    meas_obj = self.data[meas][kid][temp][atten]['sweep'][sample]

                # save data (TEMPORAL)
                self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/fit-resonators/data/{kid}')
                np.save(f'{self.work_dir}{self.project_name}/results/{meas}/fit-resonators/data/{kid}/fit-{kid}-{temp:03}-{atten:03}-S{sample}', \
                        meas_obj.fit)

                printc(f'fit-{kid}-{temp:03}-{atten:03}-S{sample} file saved', 'ok')
                
                # show fit results
                if verbose:
                    printc(f"R E S U L T S: {sweep_sample}", "title2")
                    printc(f"fr: {1e-6*fitRes[sweep_sample]['fr']:,.3f} MHz", 'info')
                    printc(f"Qr: {fitRes[sweep_sample]['Qr']:,.0f}", 'info')
                    printc(f"Qc: {fitRes[sweep_sample]['Qc']:,.0f}", 'info')
                    printc(f"Qi: {fitRes[sweep_sample]['Qi']:,.0f}", 'info')
                    printc(f"phi: {fitRes[sweep_sample]['phi']:.2f}", 'info')
                    printc(f"non: {fitRes[sweep_sample]['non']:.2f}", 'info')

                # plot fit results ?
                if plot_res:

                    # get raw data
                    frequency = meas_obj.frequency
                    s21 = meas_obj.s21

                    # get fit data
                    s21_fit = meas_obj.fit['fit_sweep']
                    
                    # difference betwwen the magnitude of s21 and s21-fit
                    #err = 20*np.log10(np.abs(s21)) - 20*np.log10(np.abs(s21_fit))

                    fig_fit, axs_fit = subplots(1, 2, num=sample, figsize=(18, 10))
                    fig_fit.subplots_adjust( top=0.98, bottom=0.075, left=0.05, right=0.98,
                                            hspace=0.0, wspace=0.12)
                    
                    axs_fit[0].plot(1e-6*frequency, 20*np.log10(np.abs(s21)), 'b.-', label='Raw data')
                    axs_fit[0].plot(1e-6*frequency, 20*np.log10(np.abs(s21_fit)), 'k-', label='Fit')
                    axs_fit[0].set_ylabel('S21 [dB]')
                    axs_fit[0].set_xlabel('Frequency [MHz]')
                    axs_fit[0].legend()

                    fr = meas_obj.fit['fr']
                    Qr = meas_obj.fit['Qr']
                    Qc = meas_obj.fit['Qc']
                    Qi = meas_obj.fit['Qi']

                    axs_fit[0].axvline(1e-6*fr, color='b', linestyle='dashed', lw=1)
                    f0_idx = np.argmin(np.abs(fr - frequency))
                    axs_fit[0].plot(1e-6*frequency[f0_idx], 20*np.log10(np.abs(s21[f0_idx])), 'Dm')

                    if self._temp_prefix == 'D':
                        temp_units = 'mK'
                        
                    elif self._temp_prefix == 'B':
                        temp_units = 'K'

                    # add dip-depth
                    dip_depth = meas_obj.get_dip_depth()

                    if 'Qr_err' in meas_obj.fit:

                        Qr_err = meas_obj.fit['Qr_err']
                        Qc_err = meas_obj.fit['Qc_err']
                        Qi_err = meas_obj.fit['Qi_err']
                        
                        axs_fit[0].text(0.05, 0.1, f'{kid}\nT: {temp[1:]}{temp_units}\nA: {atten[1:]}dB\nF0: {fr/1e6:.2f} MHz \
                            \nQr: {Qr:,.0f}+/-{Qr_err:,.0f}\nQc: {Qc:,.0f}+/-{Qc_err:,.0f} \
                            \nQi: {Qi:,.0f}+/-{Qi_err:,.0f}\nDip-depth: {dip_depth:.2f} dB', 
                            {'fontsize':12, 'color':'black'}, transform=axs_fit[0].transAxes, 
                            bbox=dict(facecolor='orange', alpha=0.5))
                    
                    else:
                        axs_fit[0].text(0.05, 0.1, f'{kid}\nT: {temp[1:]} {temp_units}\nA: {atten[1:]}dB\nF0: {fr/1e6:.2f} MHz \
                                    \nQr: {Qr:,.0f}\nQc: {Qc:,.0f}\nQi: {Qi:,.0f}\nDip-depth: {dip_depth:.2f} dB', 
                                    {'fontsize':12, 'color':'black'}, transform=axs_fit[0].transAxes, 
                                    bbox=dict(facecolor='orange', alpha=0.5))

                    axs_fit[0].grid()

                    axs_fit[1].axis('equal')
                    axs_fit[1].plot(s21.real, s21.imag, 'r.-', label='Raw data')
                    axs_fit[1].plot(s21_fit.real, s21_fit.imag, 'k-', label='Fit')
                    axs_fit[1].plot(s21[f0_idx].real, s21[f0_idx].imag, 'Dc')

                    axs_fit[1].set_xlabel('I[V]')
                    axs_fit[1].set_ylabel('Q[V]')
                    axs_fit[1].legend()
                    axs_fit[1].grid()

                    # save figures
                    name = f'{kid}-{temp}-{atten}-S{sample}'
                    self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/fit-resonators/plots/{kid}')

                    fig_fit.savefig(f'{self.work_dir}{self.project_name}/results/{meas}/fit-resonators/plots/{kid}/{name}.png')
                    close(fig_fit)

                del fitRes[sweep_sample]

    
    def clean_fit_resonators(self, kids='all', temperatures='all', attenuations='all', 
                              samples='all', meas='vna'):
        """
        Reload fits data.
        Parameters
        ----------
        kids:           [list] detectors.
        temperatures:   [list] bath/bb temperatures.
        attenuations:   [list] attenuations.
        samples:        [list] samples.
        meas:           [str] measurement system: 'vna' or 'hmd'.
        ----------
        """

        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        for kid in selected_kids:

            # get temperatures
            if temperatures == "all":
                selected_temps = np.sort(list(self.data[meas][kid].keys()))
            else:
                selected_temps = self._parse_items(temperatures, prefix=self._temp_prefix, nzeros=self._temp_zeros)

            for temperature in selected_temps:

                # get attenuations
                if attenuations == "all":
                    selected_attens = np.sort(list(self.data[meas][kid][temperature].keys()))
                else:
                    selected_attens = self._parse_items(attenuations, prefix='A', nzeros=3)

                for atten in selected_attens:

                    if samples == "all":
                        if meas == 'vna':
                            selected_samples = self.data[meas][kid][temperature][atten].keys()
                        else:
                            selected_samples = self.data[meas][kid][temperature][atten]['sweep'].keys()
                    else:
                        selected_samples = samples.copy()
                
                    for sample in selected_samples:
                        
                        # cleaning fitting
                        if meas == 'vna':
                            self.data[meas][kid][temperature][atten][sample].fit = {}
                        
                        else:
                            self.data[meas][kid][temperature][atten]['atten'][sample].fit = {}


    def load_fit_resonators(self, kids='all', meas='vna'):
        """
        Load fit resonators per kid.
        Parameters
        ----------
        kids:           [list] detectors.
        meas:           [str] measurement system: 'vna' or 'hmd'.
        ----------
        """

        # fit path
        fit_path = f'{self.work_dir}{self.project_name}/results/{meas}/fit-resonators/data/'

        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        for kid in selected_kids:

            if os.path.isdir( os.path.join(fit_path, kid) ):

                fit_files = os.listdir( os.path.join(fit_path, kid) )

                for fit_file in fit_files:                    

                    name_no_ext = fit_file.split('.npy')[0]
                    items_from_name = name_no_ext.split('-')

                    temp_item = items_from_name[2]
                    atten_item = items_from_name[3]
                    sample_item = int(items_from_name[4][1:])

                    fit_data = np.load( os.path.join(fit_path, kid, fit_file), allow_pickle=True ).item()

                    if meas == 'vna':
                        self.data[meas][kid][temp_item][atten_item][sample_item].fit = fit_data
                    else:
                        self.data[meas][kid][temp_item][atten_item]['sweep'][sample_item].fit = fit_data

            else:
                printc(f"{kid} not available.", "warn")


    def plot_sweep_kids_vs_temps(self, kids, temperatures='all', attenuations='over', 
                                samples=[0], meas='vna', include_fit=True, **kwargs):
        """
        Plot sweep of each kid at defined temperatures.
        Parameters
        ----------
        kids:           [list] detectors.
        temperatures:   [list] temperatures.
        samples:        [list] samples.
        meas:           [str] measurement system: 'vna' or 'hmd'.
        include_fit:    [bool] include fir plotting, if available.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # color map type
        cmap = kwargs.pop('cmap', 'jet')
        # ----------------------------------------------

        cmap_obj = matplotlib.cm.get_cmap(cmap)

        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
            selected_kids = np.sort(selected_kids)
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        for kid in selected_kids:

            # folder to stores images
            self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/sweeps/plots/{kid}')

            k = int(kid[1:])

            fig, axs = subplots(1, 2, num=f'{kid}', figsize=(18, 10))
            fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.125)

            # get temperatures
            if temperatures == "all":
                selected_temps = np.sort(list(self.data[meas][kid].keys()))
            else:
                selected_temps = self._parse_items(temperatures, prefix=self._temp_prefix, nzeros=self._temp_zeros)

            cnt = 0

            for temperature in selected_temps:

                # get attenuations
                if attenuations == "all":
                    selected_attens = np.sort(list(self.data[meas][kid][temperature].keys()))
                elif attenuations == "over":
                    over_atten = self.sweep_overdriven[k]
                    selected_attens = self._parse_items([over_atten], prefix='A', nzeros=3)
                else:
                    selected_attens = self._parse_items(attenuations, prefix='A', nzeros=3)

                for atten in selected_attens:

                    if float(atten[1:]) >= self.sweep_overdriven[k]:

                        if samples == "all":
                            if meas == 'vna':
                                selected_samples = self.data[meas][kid][temperature][atten].keys()
                            else:
                                selected_samples = self.data[meas][kid][temperature][atten]['sweep'].keys()
                        else:
                            selected_samples = samples.copy()
                    
                        for sample in selected_samples:

                            available_sample = self._check_availability(kid, temperature, atten, sample=sample, meas_type=meas)

                            if available_sample:

                                if meas == 'vna':
                                    meas_obj = self.data[meas][kid][temperature][atten][sample]
                                elif meas == 'hmd':
                                    meas_obj = self.data[meas][kid][temperature][atten]['sweep'][sample]

                                # get total number of samples to define color bars
                                if cnt == 0:
                                    nsamples = len(selected_samples) * len(selected_attens) * len(selected_temps)
                                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=nsamples)

                                axs[0].plot(1e-6*meas_obj.frequency, 20*np.log10(np.abs(meas_obj.s21_nr)), color=cmap_obj(norm_color(cnt)))
                        
                                axs[0].set_xlabel(f'Frequency [MHz]')
                                axs[0].set_ylabel(f'S21 [dB]')
                                axs[0].grid(True, which="both", ls="-")

                                if self._temp_prefix == 'B':
                                    current_temp = meas_obj.hdr['BLACKBOD']
                                    temp_units = 'K'
                                else:
                                    current_temp = 1e3*meas_obj.hdr['SAMPLETE']
                                    temp_units = 'mK'

                                full_str = f'{current_temp:.1f} {temp_units}, {atten[1:]} dB'

                                axs[1].plot(meas_obj.s21.real, meas_obj.s21.imag, label=full_str, color=cmap_obj(norm_color(cnt)))                                  

                                axs[1].axis('equal')
                                axs[1].set_xlabel(f'Q [V]')
                                axs[1].set_ylabel(f'I [V]')
                                axs[1].grid(True, which="both", ls="-")
                                axs[1].legend(ncols=2, fontsize=12)

                                if include_fit and meas_obj.fit != {}:
                                    axs[0].plot(1e-6*meas_obj.frequency, 20*np.log10(np.abs(meas_obj.fit['fit_sweep'])), 'k--')
                                    axs[1].plot(meas_obj.fit['fit_sweep'].real, meas_obj.fit['fit_sweep'].imag, 'k--')

                                cnt += 1

                            else:
                                printc(f"Data sample {kid}-{temperature}-{atten}-{sample} not available.", "warn")

            # save plots
            fig.savefig(f'{self.work_dir}{self.project_name}/results/{meas}/sweeps/plots/{kid}/{kid}-{temperature}-{atten}-{sample}-sweep.png')


    # get drive power vs fit
    def plot_power_vs_qs(self, kids="all", temperatures="all", meas='vna', sample=0, 
                         flag_kids=[]):
        """
        Plot drive power vs Qi and Qc factors for the given detectors at a given temperature.
        Parameters
        ----------
        kids:                   [int/str/list] detectors.
        temperatures:           [int/str/list] base/bb temperature.
        meas:                   [str] measurement system.
        sample:                 [int] sample number.
        flag_kids:              [list] masked detectors.
        ----------
        """

        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        # flagged kids
        if len(flag_kids) > 0:
            flag_selected_kids = self._parse_items(flag_kids, prefix='K', nzeros=3)
        else:
            flag_selected_kids = []

        # get overdriven attenuations
        if len(self.sweep_overdriven) == 0:
            ovr_attens = np.zeros(len(selected_kids))
        else:
            ovr_attens = self.sweep_overdriven.copy()

        # get warm attenuation
        try:
            input_atten = self.data[meas]['header']['ATT_UC'] + \
                      self.data[meas]['header']['ATT_C'] + \
                      self.data[meas]['header']['ATT_RT']
            
        except:
            input_atten = 0

        # get figure and axis
        fig_temps, axs_temps = [], []
        fig_fr_temps, axs_fr_temps = [], []

        temperature_names = []

        qs_results = {}

        lstyles = ['solid', 'dashed', 'dotted']

        for kid in selected_kids:
            
            if 'K' in kid and not kid in flag_selected_kids:

                # get temperatures
                if temperatures == "all":
                    selected_temps = self.data[meas][kid].keys()
                else:
                    selected_temps = self._parse_items(temperatures, prefix=self._temp_prefix, nzeros=self._temp_zeros) 

                for t, temperature in enumerate(selected_temps):

                    if not temperature in temperature_names:

                        # get figures per each temperature
                        fig, axs = subplots(1, 2, num=f'{temperature}', figsize=(20, 12))
                        fig.subplots_adjust( top=0.96, bottom=0.075, left=0.060, right=0.98, hspace=0.0, wspace=0.15 )

                        fig_temps.append(fig)
                        axs_temps.append(axs)

                        fig_fr, axs_fr = subplots(1, 1, num=f'f0s-{temperature}', figsize=(10, 12))
                        fig_fr.subplots_adjust( top=0.96, bottom=0.075, left=0.060, right=0.98, hspace=0.0, wspace=0.15 )

                        fig_fr_temps.append(fig_fr)
                        axs_fr_temps.append(axs_fr)

                    temperature_names.append(temperature)

                    # get detector number
                    k = int(kid[1:])
                    qis, qcs, frs = [], [], []
                    num_attens = []

                    # get attenuations
                    attenuations_numbers = sorted([float(a[1:]) for a in self.data[meas][kid][temperature].keys()])
                    selected_attens = [f'A{a:.1f}' for a in attenuations_numbers]

                    for atten in selected_attens:

                        # get overdriven attenuations
                        num_atten = float(atten[1:])

                        if num_atten >= ovr_attens[k]:

                            if meas == 'vna':
                                meas_obj = self.data[meas][kid][temperature][atten][sample]
                            
                            else:
                                meas_obj = self.data[meas][kid][temperature][atten]['sweep'][sample]

                            if meas_obj.fit != {}:
                                # get fit info
                                qi = meas_obj.fit['Qi']
                                qc = meas_obj.fit['Qc']
                                fr = meas_obj.fit['fr']

                                qis.append(qi)
                                qcs.append(qc)
                                frs.append(fr)

                                num_attens.append(-1*(num_atten + self.add_input_atten + input_atten))

                    qis = np.array(qis)
                    qcs = np.array(qcs)
                    frs = np.array(frs)

                    if not kid in qs_results:
                        qs_results[kid] = {}

                    if not temperature in qs_results[kid]:
                        qs_results[kid][temperature] = {}

                    qs_results[kid][temperature]['qi'] = qis
                    qs_results[kid][temperature]['qc'] = qcs
                    qs_results[kid][temperature]['fr'] = frs
                    qs_results[kid][temperature]['att'] = num_attens

                    lstyle = lstyles[int(k/10)]

                    axs_temps[t][0].plot(num_attens, 1e-3*qis, 's-', lw=1.75, linestyle=lstyle)
                    axs_temps[t][0].set_xlabel('Drive power [dBm]')
                    axs_temps[t][0].set_ylabel('Qi [k]')
                    axs_temps[t][0].set_title(f'Qi')
                    axs_temps[t][0].grid(True, which="both", ls="-")

                    axs_temps[t][1].plot(num_attens, 1e-3*qcs, 's-', lw=1.75, label=f'{kid} {1e-9*np.mean(frs):.2f} GHz {num_attens[0]} dB', linestyle=lstyle)
                    axs_temps[t][1].set_xlabel('Drive power [dBm]')
                    axs_temps[t][1].set_ylabel('Qc [k]')
                    axs_temps[t][1].set_title(f'Qc')
                    axs_temps[t][1].grid(True, which="both", ls="-")
                    axs_temps[t][1].legend(ncols=2, fontsize=12)

                    axs_fr_temps[t].plot(num_attens, 1e6*(frs-frs[0])/frs[0], 's-', lw=1.75, linestyle=lstyle, label=f'{kid} {num_attens[0]} dB')
                    axs_fr_temps[t].set_xlabel('Drive power [dBm]')
                    axs_fr_temps[t].set_ylabel('ffs [ppm]')
                    axs_fr_temps[t].set_title(f'Resonance frequency')
                    axs_fr_temps[t].grid(True, which="both", ls="-")
                    axs_fr_temps[t].legend(ncols=2, fontsize=12)

                    opt_info = 'Dark test'
                    if self._temp_prefix == 'B':
                        opt_info = f'Opt test, BB:{meas_obj.hdr['BLACKBOD']}'

                    axs_temps[t][0].text(0.05, 0.85, f'{opt_info}\nBase temperature: {meas_obj.hdr['SAMPLETE']*1e3:.1f} mK\nFixed input atten: {self.add_input_atten+input_atten} dB',
                                {'fontsize':12, 'color':'black'}, transform=axs_temps[t][0].transAxes, 
                                bbox=dict(facecolor='orange', alpha=0.5))
                    
                    axs_fr_temps[t].text(0.05, 0.85, f'{opt_info}\nBase temperature: {meas_obj.hdr['SAMPLETE']*1e3:.1f} mK\nFixed input atten: {self.add_input_atten+input_atten} dB',
                                {'fontsize':12, 'color':'black'}, transform=axs_temps[t][0].transAxes, 
                                bbox=dict(facecolor='orange', alpha=0.5))

        # save results
        self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/drive_power-qs/data')

        # save figures
        self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/drive_power-qs/plots')
        
        for i in range(len(fig_temps)):
            
            fig_temps[i].savefig(f'{self.work_dir}{self.project_name}/results/{meas}/drive_power-qs/plots/qs-{temperature_names[i]}.png')
            fig_fr_temps[i].savefig(f'{self.work_dir}{self.project_name}/results/{meas}/drive_power-qs/plots/f0-{temperature_names[i]}.png')

            np.save(f'{self.work_dir}{self.project_name}/results/{meas}/drive_power-qs/data/{temperature_names[i]}', qs_results)


    def qs_summary(self, temperature, kids="all", attenuations="all", samples="all",
                   flag_kids=[], meas="vna"):
        """
        Get and show Qs results for a given temperature.
        Parameters
        ----------
        temperature:    [int/str] temperature.
        kids:           [list/str] KID ID. If 'all' it will take all the resonators.
        attens:         [list/str] attenuation. If 'all' it will select all the attenuations.
        samples:        [list/str] samples. If 'all', select all the samples.
        flag_kids:      [list] flag kids.
        meas:           [str] measurement system: 'vna' or 'hmd', 'vna' by default.
        ----------
        """

        # get temperature in KID-analiser format
        temperature = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)

        # get kids
        if kids == "all":
            selected_kids = [kid for kid in self.data[meas].keys() if kid.startswith('K')]
        else:
            selected_kids = self._parse_items(kids, prefix='K', nzeros=3) 

        if len(flag_kids) > 0:
            flag_selected_kids = self._parse_items(flag_kids, prefix='K', nzeros=3)
        else:
            flag_selected_kids = []

        # get overdriven attenuations
        if len(self.sweep_overdriven) == 0:
            over_attens = np.zeros(len(selected_kids))
        else:
            over_attens = self.sweep_overdriven.copy()

        max_qis, mean_qcs, fr_from_qis = [], [], []
        max_qis_err, mean_qcs_err = [], []

        qs_summary = {}

        for kid in selected_kids:

            k = int(kid[1:])

            if not kid in flag_selected_kids:

                # get attenuations
                if attenuations == "all":
                    attenuations_numbers = sorted([float(a[1:]) for a in self.data[meas][kid][temperature].keys()])
                    selected_attens = [f'A{a:.1f}' for a in attenuations_numbers]
                else:
                    selected_attens = self._parse_items(attenuations, prefix='A', nzeros=3)

                none_cnt = 0
                qi_atten_c, qc_atten_c, fr_atten_c = [], [], []

                for atten in selected_attens:

                    if samples == "all":
                        if meas == 'vna':
                            selected_samples = self.data[meas][kid][temperature][atten].keys()
                        else:
                            selected_samples = self.data[meas][kid][temperature][atten]['sweep'].keys()
                    else:
                        selected_samples = samples.copy()
                
                    qi_sample, qc_sample, fr_sample = [], [], []

                    for sample in selected_samples:

                        # get overdriven attenuations
                        num_atten = float(atten[1:])

                        if num_atten >= over_attens[k]:

                            # get fit info
                            if meas == 'vna':
                                meas_obj = self.data[meas][kid][temperature][atten][sample]

                            elif meas == 'hmd':
                                meas_obj = self.data[meas][kid][temperature][atten]['sweep'][sample]

                            meas_fit = meas_obj.fit

                            try:
                                # get Qi, Qc and fr
                                qi = meas_fit['Qi']
                                qc = meas_fit['Qc']
                                fr = meas_fit['fr']
                            
                            except:
                                qi = np.nan
                                qc = np.nan
                                fr = np.nan

                            qi_sample.append(qi)
                            qc_sample.append(qc)
                            fr_sample.append(fr)
                   
                    if len(qi_sample) > 0:
                        qi_atten_c.append(qi_sample)
                        qc_atten_c.append(qc_sample)
                        fr_atten_c.append(fr_sample)

                    else:
                        none_cnt += 1

                qi_mean_atten = np.mean(qi_atten_c, axis=1)
                qc_mean_atten = np.mean(qc_atten_c, axis=1)
                fr_mean_atten = np.mean(fr_atten_c, axis=1)

                qi_std_atten = np.std(qi_atten_c, axis=1)
                qc_std_atten = np.std(qc_atten_c, axis=1)

                # qi
                max_qi_idx = np.nanargmax(qi_mean_atten)
                max_qi_per_kid = qi_mean_atten[max_qi_idx]
                max_qi_per_kid_err = qi_std_atten[max_qi_idx]

                # qc
                mean_qc_per_kid = np.nanmean(qc_mean_atten)

                N, qc_err = len(qc_mean_atten), 0
                for a in range(N):
                    qc_err += (qc_mean_atten[a] - mean_qc_per_kid)**2
                
                mean_qc_per_kid_err = np.sqrt(qc_err/(N-1))

                # collect all of them
                max_qis.append( max_qi_per_kid )
                max_qis_err.append( max_qi_per_kid_err )

                mean_qcs.append( mean_qc_per_kid )
                mean_qcs_err.append( mean_qc_per_kid_err )
                fr_from_qis.append( fr_mean_atten[max_qi_idx] )

                max_qi_atten = selected_attens[max_qi_idx + none_cnt]

                dip_depth = self.data[meas][kid][temperature][max_qi_atten][sample].get_dip_depth()

                fr_at_max_qi = fr_mean_atten[max_qi_idx]

                printc( f'{kid}\t{max_qi_atten}\tf0: {fr_at_max_qi*1e-6:2f} MHz\tQi:{1e-3*max_qi_per_kid:.2f} k\tQc: {1e-3*mean_qc_per_kid:.2f} k\tDip-depth: {dip_depth:.2f} dB', 'title2')

            else:

                max_qi_per_kid, mean_qc_per_kid = None, None

                a = self._parse_single_item(over_attens[k], prefix='A', nzeros=3)  
                
                max_qi_atten = a

                # get fit info
                if meas == 'vna':
                    meas_obj = self.data[meas][kid][temperature][a][sample]

                elif meas == 'hmd':
                    meas_obj = self.data[meas][kid][temperature][a]['sweep'][sample]

                # get f0
                fr_at_max_qi = meas_obj.hdr['F0']

                # get dip depth
                dip_depth = meas_obj.get_dip_depth()

                print( f'{kid} \t{max_qi_atten}\tf0: {fr_at_max_qi*1e-6:2f} MHz\tQi: no fit\tQc: no fit\tDip-depth: {dip_depth:.2f} dB')

            qs_summary[kid] = {}
            qs_summary[kid] = [float(max_qi_atten[1:]), fr_at_max_qi, max_qi_per_kid, mean_qc_per_kid, dip_depth]

        printc(f'-------- Q I --------', 'info')
        printc(f'From {np.min(max_qis)*1e-3:.2f} k to {np.max(max_qis)*1e-3:.2f} k', 'title1')
        printc(f'Qis mean: {np.mean(max_qis)*1e-3:.2f} k', 'title1')
        printc(f'Qis median: {np.median(max_qis)*1e-3:.2f} k', 'title1')
        printc(f'Qis std: {np.std(max_qis)*1e-3:.2f} k', 'title1')
        
        printc(f'-------- Q C --------', 'info')
        printc(f'From {np.min(mean_qcs)*1e-3:.2f} k to {np.max(mean_qcs)*1e-3:.2f} k', 'title1')
        printc(f'Qcs mean: {np.mean(mean_qcs)*1e-3:.2f} k', 'title1')
        printc(f'Qcs median: {np.median(mean_qcs)*1e-3:.2f} k', 'title1')
        printc(f'Qcs std: {np.std(mean_qcs)*1e-3:.2f} k', 'title1')

        # save results
        self._create_diry(f'{self.work_dir}{self.project_name}/results/{meas}/')
        np.save(f'{self.work_dir}{self.project_name}/results/{meas}/sweep-summary', qs_summary)

        return max_qis, mean_qcs, fr_from_qis


    def find_overdriven_atts(self, temperature, sample=0, non_linear_thresh=0.7, **kwargs):
        """
        Find the pre-overdriven attenuations given the fit results + manual selection.
        Parameters
        ----------
        temperature:        [int/str] temperature. If 'None' it will take all the temperatures.
        sample:             [int] sample number. If 'None' take all the samples/repeats.
        non_linear_thresh:  [float] non-linearity threshold to define an overdriven state.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Number of columns
        num_cols = kwargs.pop('num_cols', 5)
        # Number of columns
        num_rows = kwargs.pop('num_rows', 6)
        # ----------------------------------------------

        # select all the detectors
        kids = [kid for kid in self.data['vna'].keys() if kid.startswith('K')]
        # get temperatures
        temperature = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)

        # init overdriven atts at zero
        over_attens_by_temp = np.zeros_like(kids, dtype=float)

        cnt = 0
        fig_cnt, total_cnt = -1, 0
        number_fig_over = 0

        for k, kid in enumerate(kids):

            # select all attenuations
            count_atts = 0
            attenuations_numbers = sorted([float(a[1:]) for a in self.data['vna'][kid][temperature].keys()])
            attenuations = [f'A{a:.1f}' for a in attenuations_numbers]

            nonlinear_params_per_kid = np.zeros_like(attenuations, dtype=float)

            for a, attenuation in enumerate(attenuations):
                
                try:
                    # get non-linearity param
                    nonlinear_params_per_kid[a] = self.data['vna'][kid][temperature][attenuation][sample].fit['non']
                
                except:
                    count_atts += 1

            if count_atts < len(attenuations):
                
                idx = len(nonlinear_params_per_kid)-np.where(nonlinear_params_per_kid[::-1]>non_linear_thresh)[0] - 1
                
                if len(idx) > 0:
                    idx = idx[0]
                else:
                    idx = 0

                text_size = 14
            
            else:

                idx = int(len(attenuations)/2)
                num_cols = len(attenuations)

                text_size = 9

            if k%num_rows == 0:

                if fig_cnt >= 0:
                    overdriven_figure = overdrivenFigure(fig, ax, over_attens_by_temp, 
                                                         over_atts_mask, over_atts_mtx, number_fig_over,
                                                         num_cols=num_cols, num_rows=num_rows)
                    
                    self.sweep_overdriven = overdriven_figure.over_attens_by_temp

                fig, ax = subplots(num_rows, num_cols)
                fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.0, wspace=0.0)
                            
                over_atts_mtx = np.zeros((num_rows, num_cols))
                over_atts_mask = np.zeros((num_rows, num_cols), dtype=bool)

                fig_cnt += 1
                number_fig_over = fig_cnt
                cnt = 0

            # assign overdriven attenuations
            over_attens_by_temp[k] = float(attenuations[idx][1:])

            for i in np.arange(num_cols):

                idx_map = i + idx - int(num_cols/2)

                ii = cnt%num_cols
                jj = int(cnt/num_cols)

                if idx_map >= 0 and idx_map<len(attenuations):

                    # get s21 data
                    s21 = self.data['vna'][kid][temperature][attenuations[idx_map]][sample].s21
                    
                    ax[jj, ii].plot(s21.real, s21.imag, 'r.-')
                    ax[jj, ii].tick_params(axis='x', labelsize=text_size)
                    ax[jj, ii].tick_params(axis='y', labelsize=text_size)

                    # if there is a fit, display it
                    if self.data['vna'][kid][temperature][attenuations[idx_map]][sample].fit != {}:

                        fit_s21 = self.data['vna'][kid][temperature][attenuations[idx_map]][sample].fit['fit_sweep']
                        
                        ax[jj,ii].plot(fit_s21.real, fit_s21.imag, 'k')
                
                    ax[jj,ii].axis('equal')

                    if i == int(num_cols/2):
                        ax[jj,ii].patch.set_facecolor('green')
                    else:
                        ax[jj,ii].patch.set_facecolor('blue')

                    ax[jj,ii].patch.set_alpha(0.2)
                    ax[jj,ii].text(0.2, 0.1, attenuations[idx_map] + ' dB', \
                                    {'fontsize': text_size, 'color':'white'}, \
                                    bbox=dict(facecolor='purple', alpha=0.95), \
                                    transform=ax[jj,ii].transAxes)
                    
                    over_atts_mtx[jj,ii] = float(attenuations[idx_map][1:])
                    over_atts_mask[jj,ii] = True

                if jj == num_cols or total_cnt == len(kids)-1:
                    ax[jj,ii].set_xlabel("I [V]")

                if cnt%num_cols == 0:
                    ax[jj,ii].set_ylabel(kid+"\nQ [V]")

                cnt += 1
                total_cnt += 1

        overdriven_figure = overdrivenFigure(fig, ax, over_attens_by_temp, 
                         over_atts_mask, over_atts_mtx, number_fig_over,
                         num_cols=num_cols, num_rows=num_rows)
        
        self.sweep_overdriven = overdriven_figure.over_attens_by_temp


    def merge_single_psd_meas(self, kid, temperature, attenuation, samples=[0, 1], save_figs=True,
                              df_method='phase-inter', tune='on', trim_psd_edges=[3, -20], 
                              plot_phase=False, plot_streams=False, despike=True, mask=None, 
                              ffs_units=False, binning_points=300, sweep_sample=0, **kwargs):
        """
        Merge single PSD measurement.
        Parameters
        ----------
        kid:            [int/str] detector id.
        temperature:    [int/str] temperature.
        attenuation:    [int/str] attenuation.
        df_method:      [str] Frequency shift calculation method.
        samples:        [int/str] samples to merge.
        tune:           [str] 'on'/'off' resonace sample.
        trim_psd_edges: [list] define indexes to trim edges of each psd.
                        Apply it to ignore data affected by aliasing.
        plot_phase:     [bool] display the phase?
        plot_streams:   [bool] display the timestreams?
        despike:        [bool] despike streams?
        mask:           [bool] apply auto masking to select timestream samples?
        ffs_units:      [bool] get psd in ffs units?
        binning_points: [int] number of points of the merged psd.
        sweep_sample:   [int] sample sweep.
        ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Linear fit for the phase
        linear_fit = kwargs.pop('linear_fit', True)
        # f0 threshold for lineal fit
        f0_thresh_linear_fit = kwargs.pop('f0_thresh_linear_fit', 1e4)
        # f0 threshold for phase modeling
        f0_phase_thresh = kwargs.pop('f0_phase_thresh', 25e4)
        # f0 threshold for
        smooth_params_phase = kwargs.pop('smooth_params_phase', [31, 3])
        # window size for despiking
        win_size = kwargs.pop('win_size', 300)
        # sigma threshold for auto masking
        sigma_thresh_masking = kwargs.pop('sigma_thresh_masking', 3)
        # despiking sigma threshold
        despike_sigma_thresh = kwargs.pop('despike_sigma_thresh', 4)
        # frequency limits to average psd for automasking
        freq_avg_psd = kwargs.pop('freq_avg_psd', [4, 20])
        # Min number of points to be consider a source
        source_min_size = kwargs.pop('source_min_size', None)
        # number of points to average to replace a single glitch event
        trim_glitch_sample = kwargs.pop('trim_glitch_sample', 3)
        # fraction of window size at the edges to define the noise sample
		# which stats will replace a glitch event.
        win_noise_sample = kwargs.pop('win_noise_sample',  0.25)
        # verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        # parse key format
        k = self._parse_single_item(kid, prefix='K', nzeros=3)                                          # get kid
        t = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)     # get temperature
        a = self._parse_single_item(attenuation, prefix='A', nzeros=3)                                  # get attenuation

        if samples == 'all':
            samples = list(self.data['hmd'][k][t][a]['on'].keys())

        # mask and deglitch timestreams
        for sample in samples:

            # apply auto masking
            if mask == None:
                self.data['hmd'][k][t][a][tune][sample].auto_masking(sigma_thresh=sigma_thresh_masking, 
                                                                     freq_avg_psd=freq_avg_psd)

            # deglitch timestreams
            if despike:
                self.data['hmd'][k][t][a][tune][sample].despike(win_size=win_size, sigma_thresh=despike_sigma_thresh, 
                                                                verbose=verbose, source_min_size=source_min_size,
                                                                win_noise_sample=win_noise_sample, 
                                                                trim_glitch_sample=trim_glitch_sample)

            # show timestreams?
            if plot_streams:

                for i in range(self.data['hmd'][k][t][a][tune][sample].samples):

                    fig_ts, axs_ts = subplots(2, 1, figsize=(18, 10), num=f'{k}-{t}-{a}-{sample} {tune}', sharex=True)
                    fig_ts.subplots_adjust(top=0.97, left=0.15, right=0.97, bottom=0.1, hspace=0.1)

                    if self.data['hmd'][k][t][a][tune][sample].mask[i]:
                        
                        axs_ts[0].patch.set_facecolor('green')
                        axs_ts[0].patch.set_alpha(0.2)

                        axs_ts[1].patch.set_facecolor('green')
                        axs_ts[1].patch.set_alpha(0.2)

                    else:

                        axs_ts[0].patch.set_facecolor('red')
                        axs_ts[0].patch.set_alpha(0.2)

                        axs_ts[1].patch.set_facecolor('red')
                        axs_ts[1].patch.set_alpha(0.2)

                    axs_ts[0].plot(self.data['hmd'][k][t][a][tune][sample].time_sample, self.data['hmd'][k][t][a][tune][sample].I[i])
                    axs_ts[0].set_ylabel(f'I[V]')
                    axs_ts[0].grid(True, which="both", ls="-")
                    
                    axs_ts[1].plot(self.data['hmd'][k][t][a][tune][sample].time_sample, self.data['hmd'][k][t][a][tune][sample].Q[i])
                    axs_ts[1].set_xlabel(f'Time [s]')
                    axs_ts[1].set_ylabel(f'Q[V]')
                    axs_ts[1].grid(True, which="both", ls="-")

                    fs = self.data['hmd'][k][t][a][tune][sample].hdr['SAMPLERA']
                    axs_ts[1].text(0.05, 0.1, f'{k}-{t}-{a}-{sample} {tune}\nfs: {fs:.1f} Hz\ntime: {self.data['hmd'][k][t][a][tune][sample].time_sample[-1]:.1f} s',
                                {'fontsize':12, 'color':'black'}, transform=axs_ts[1].transAxes, 
                                bbox=dict(facecolor='orange', alpha=0.5))

                    if save_figs:

                        self._create_diry(f'{self.work_dir}{self.project_name}/results/hmd/noise/timestreams/plots/{k}/{tune}')
                        fig_ts.savefig(f'{self.work_dir}{self.project_name}/results/hmd/noise/timestreams/plots/{k}/{tune}/{t}-{a}-S{sample}-repeat_{i}.png')

                        close(fig_ts)

            # calculate the frequency shift
            self.compute_single_df(kid, temperature, attenuation, sample=sample, method=df_method, ffs=ffs_units,
                            tune=tune, linear_fit=linear_fit, f0_thresh_linear_fit=f0_thresh_linear_fit,
                            smooth_params_phase=smooth_params_phase, sweep_sample=sweep_sample, 
                            f0_phase_thresh=f0_phase_thresh)
            
            # calculate the psd 
            self.compute_single_psd(kid, temperature, attenuation, sample=sample, tune=tune)

        if not k in self.psd_params:
            self.psd_params[k] = {}
        
        if not t in self.psd_params[k]:
            self.psd_params[k][t] = {}

        if not a in self.psd_params[k][t]:
            self.psd_params[k][t][a] = {}
        
        if not tune in self.psd_params[k][t][a]:
            self.psd_params[k][t][a][tune] = {}

        base_temperature = self.data['hmd'][k][t][a][tune][samples[0]].hdr['SAMPLETE']
        bb_temperature = self.data['hmd'][k][t][a][tune][samples[0]].hdr['BLACKBOD']

        updated_params = {
            'base_temperature': base_temperature,
            'bb_temperature': bb_temperature,
            'samples':  samples,
            'df_method': df_method,
            'trim_psd_edges': trim_psd_edges,
            'plot_phase':   plot_phase,
            'plot_streams': plot_streams,
            'despike':  
                {
                    'active': despike,
                    'win_size': win_size,
                    'despike_sigma_thresh': despike_sigma_thresh,
                },
            'ffs_units': ffs_units,
            'binning_points':   binning_points,
            'sweep_sample': sweep_sample,
            'mask':
                { 
                    'active': mask,
                    'sigma_thresh_masking': sigma_thresh_masking,
                    'freq_avg_psd': freq_avg_psd,
                    'source_min_size': source_min_size,
                    'trim_glitch_sample': trim_glitch_sample,
                    'win_noise_sample': win_noise_sample,
                    'samples_masked': [self.data['hmd'][k][t][a][tune][i].mask for i in self.data['hmd'][k][t][a][tune].keys()]
                },
            'f0_phase_thresh': f0_phase_thresh,
            'linear_fit':   linear_fit,
            'f0_thresh_linear_fit': f0_thresh_linear_fit,
            'smooth_params_phase': smooth_params_phase
        }

        self.psd_params[k][t][a][tune] = updated_params

        # stack all the samples
        freq_samples, psd_samples = [], []

        for sample in samples:

            f_sample, psd_sample = self.data['hmd'][k][t][a][tune][sample].psd

            if not len(psd_sample) == np.count_nonzero(np.isnan(psd_sample)):

                freq_samples.append(f_sample[trim_psd_edges[0]:trim_psd_edges[1]])
                psd_samples.append(psd_sample[trim_psd_edges[0]:trim_psd_edges[1]])
        
        # show phase calculation?
        if plot_phase:

            phase = self.data['hmd'][k][t][a]['sweep'][0].phase

            fig, axs = subplots(1, 2, num=f'{k}-{t}-{a}-{tune}-{df_method}', figsize=(18, 10))
            fig.subplots_adjust( top=0.98, bottom=0.080, left=0.065, right=0.97, wspace=0.16)

            # plot phase
            freq_phase = self.data['hmd'][k][t][a]['sweep'][sweep_sample].frequency

            f0_for_phase = self.data['hmd'][k][t][a]['on'][sweep_sample].hdr['F0']
            f0_idx = np.argmin( np.abs( freq_phase - f0_for_phase ) )

            axs[0].axhline(0, color="k", lw="1.5")
            axs[0].axvline(1e-9*freq_phase[f0_idx], color="k", lw="1.5")

            axs[0].plot(1e-9*freq_phase, phase, '.-')

            axs_phase = axs[0].inset_axes([0.08, 0.08, 0.25, 0.25])

            from_idx = f0_idx - 10
            if from_idx < 0:
                from_idx = 0

            to_idx = f0_idx + 10
            if to_idx >= len(phase):
                to_idx = len(phase) - 1

            axs_phase.axhline(0, color="k", lw="1.5")
            axs_phase.axvline(1e-9*freq_phase[f0_idx], color="k", lw="1.5")

            axs_phase.plot(1e-9*freq_phase[from_idx:to_idx], phase[from_idx:to_idx], '.-')

            if self.data['hmd'][k][t][a]['sweep'][sweep_sample].phase_mdl != None:
                axs[0].plot(1e-9*self.data['hmd'][k][t][a]['sweep'][sweep_sample].phase_mdl(phase), phase, 'r.-')
                axs_phase.plot(1e-9*self.data['hmd'][k][t][a]['sweep'][sweep_sample].phase_mdl(phase)[from_idx:to_idx], phase[from_idx:to_idx], 'r.-')

            axs[0].set_xlabel(f'Frequency [GHz]')
            axs[0].set_ylabel(f'Phase [rad]')
            axs[0].grid(True, which="both", ls="-")
            
            axs_phase.grid(True, which="both", ls="-")
            axs_phase.xaxis.set_tick_params(labelsize=11)
            axs_phase.yaxis.set_tick_params(labelsize=11)

            axs[0].xaxis.set_tick_params(labelsize=15)
            axs[0].yaxis.set_tick_params(labelsize=15)

            xc = self.data['hmd'][k][t][a]['sweep'][sweep_sample].xc.copy()
            yc = self.data['hmd'][k][t][a]['sweep'][sweep_sample].yc.copy()
            theta = self.data['hmd'][k][t][a]['sweep'][sweep_sample].theta.copy()

            # plot IQ circle
            axs[1].plot(1e3*self.data['hmd'][k][t][a]['sweep'][sweep_sample].s21.real, 
                1e3*self.data['hmd'][k][t][a]['sweep'][sweep_sample].s21.imag, '.-')
            
            # show the samples
            max_phases, min_phases = [], []
            for i in range(self.data['hmd'][k][t][a][tune][sweep_sample].samples):
                
                I_ts_derot, Q_ts_derot = derot_circle(self.data['hmd'][k][t][a][tune][sweep_sample].I[i],
                                                      self.data['hmd'][k][t][a][tune][sweep_sample].Q[i],
                                                      xc, yc, theta)

                axs[1].plot(1e3*I_ts_derot, 1e3*Q_ts_derot, 'r.')

                phase_stream = np.arctan(Q_ts_derot/I_ts_derot)
                
                max_phases.append(np.max(phase_stream))
                min_phases.append(np.min(phase_stream))

            I0_derot, Q0_derot = derot_circle(self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['IF0'], 
                                             self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['QF0'],
                                             xc, yc, theta)
                       
            axs[1].plot(1e3*I0_derot, 1e3*Q0_derot, 'ks')

            if tune == 'on':

                mean_I = np.mean(1e3*I0_derot)

                max_phase = np.max(max_phases)
                min_phase = np.min(min_phases)

                axs[1].plot([0, 1.2*mean_I], [0, 1.2*mean_I*np.tan(max_phase)], color='k', linestyle='dashed', lw=1)
                axs[1].plot([0, 1.2*mean_I], [0, 1.2*mean_I*np.tan(min_phase)], color='k', linestyle='dashed', lw=1)

                axs_phase.axhline(min_phase, color="k", lw=1, linestyle='dashed')
                axs_phase.axhline(max_phase, color="k", lw=1, linestyle='dashed')

                axs[0].axhline(min_phase, color="k", lw=1, linestyle='dashed')
                axs[0].axhline(max_phase, color="k", lw=1, linestyle='dashed')

            axs[1].axhline(0, color="k", lw="1.5")
            axs[1].axvline(0, color="k", lw="1.5")

            axs[1].set_xlabel(f'I[mV]')
            axs[1].set_ylabel(f'Q[mV]')
            axs[1].axis('equal')
            axs[1].grid(True, which="both", ls="-")

            self._create_diry(f'{self.work_dir}{self.project_name}/results/hmd/sweep/plots/{k}')
            fig.savefig(f'{self.work_dir}{self.project_name}/results/hmd/sweep/plots/{k}/sweep-{k}-{t}-{a}-{tune}.png')

        # merge the samples
        f_merge, psd_merge = merge_spectra(freq_samples, psd_samples, n_pts=binning_points)

        # save average psd 
        self._create_diry(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}')
        np.save(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}/psd_{tune}-{k}-{t}-{a}-S{sample}', 
                self.data['hmd'][k][t][a][tune][sample].psd)

        # save psd_params
        np.save(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/analysis_params', self.psd_params)

        # save merge data
        np.save(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}/merge-psd_{tune}-{k}-{t}-{a}-S{sample}', 
                [f_merge, psd_merge])

        printc(f'psd {tune} {kid}-{temperature:03}-{attenuation:03}-S{sample} file saved', 'ok')

        return f_merge, psd_merge


    def on_off_psd_resonance(self, kid, temperature, attenuation, samples_on=[0, 1], samples_off=[0, 1],
                             psd_params={}, fit_psd=True, psd_to_fit='on-off', **kwargs):
        """
        Get on/off psd resonance, as well as the difference.
        Parameters
        ----------
        kid:            [int/str] detector id.
        temperature:    [int/str] temperature.
        attenuation:    [int/str] attenuation.
        samples:        [int/str] samples to merge.
        psd_params:     [dict] psd calculation parameters.
        ----------
        """

        # Key arguments
        # ----------------------------------------------
        # verbose
        verbose = kwargs.pop('verbose', False)
        # save psd figure?
        save_psd_figure = kwargs.pop('save_psd_figure', True)
        # ----------------------------------------------

        if fit_psd:
            assert psd_to_fit in ['on', 'on-off'], "PSD to fit not valid.\nTry 'on' or 'off' resonance."

        # parse key format
        k = self._parse_single_item(kid, prefix='K', nzeros=3)                                          # get kid
        t = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)     # get temperature
        a = self._parse_single_item(attenuation, prefix='A', nzeros=3)                                  # get attenuation

        if samples_on == "all":
            samples_on = list(self.data['hmd'][k][t][a]['on'].keys())

        if samples_off == "all":
            samples_off = list(self.data['hmd'][k][t][a]['off'].keys())

        # on resonance
        freq_res_on, psd_res_on = self.merge_single_psd_meas(k, t, a, samples=samples_on, 
                                                             tune='on', **psd_params, verbose=verbose)

        # off resonance
        freq_res_off, psd_res_off = self.merge_single_psd_meas(k, t, a, samples=samples_off, 
                                                               tune='off', **psd_params, verbose=verbose)

        psd_on_off = [i if i>0 else np.nan for i in psd_res_on-psd_res_off]

        if save_psd_figure:

            fig_psd, axs_psd = subplots(1, 1, num=f'PSD-{k}-{t}-{a}', figsize=(18, 10))
            fig_psd.subplots_adjust( top=0.95, bottom=0.075, left=0.075, right=0.98,
                                            hspace=0.0, wspace=0.12)

            colors = ['r', 'b']
            for i, mode in enumerate(['on', 'off']):
                on_samples = len(self.data['hmd'][k][t][a][mode])
                for s in range(on_samples):
                    meas_on = self.data['hmd'][k][t][a][mode][s]
                    axs_psd.loglog(meas_on.psd[0][3:], meas_on.psd[1][3:], alpha=0.3, color=colors[i])

            axs_psd.loglog(freq_res_on, psd_res_on, 'r', lw=2, label='on')
            axs_psd.loglog(freq_res_off, psd_res_off, 'b', lw=2, label='off')

            axs_psd.loglog(freq_res_on, psd_on_off, 'k', lw=2, label='on-off')

            axs_psd.grid(True, which="both", ls="-")
            axs_psd.set_xlim([freq_res_on[0], freq_res_on[-1]])
            #axs_psd.set_title(f'PSD')
            axs_psd.set_xlabel(f'Frequency [Hz]')
            axs_psd.set_ylabel(f'PSD df [Hz²/Hz]')
            axs_psd.legend()

            # add the sweep and on/off resonance
            iq_circle = axs_psd.inset_axes([0.08, 0.08, 0.25, 0.25])

            meas_sweep = self.data['hmd'][k][t][a]['sweep'][0]

            xc = meas_sweep.xc.copy()
            yc = meas_sweep.yc.copy()
            theta = meas_sweep.theta.copy()

            meas_on_stream = self.data['hmd'][k][t][a]['on'][0]
            meas_off_stream = self.data['hmd'][k][t][a]['off'][0]

            iq_circle.plot(meas_sweep.s21.real, meas_sweep.s21.imag, 'k')

            # on-resonance
            for i in range(len(meas_on_stream.I)):
                # check mask
                if self.data['hmd'][k][t][a]['on'][0].mask[i]:
                    # rotate timestreams
                    I_on = meas_on_stream.I[i]
                    Q_on = meas_on_stream.Q[i]
                    
                    I_on_derot, Q_on_derot = derot_circle(I_on, Q_on, xc, yc, theta)

                    iq_circle.plot(I_on_derot, Q_on_derot, 'r.')

            # off-resonance
            for i in range(len(meas_off_stream.I)):
                # check mask
                if self.data['hmd'][k][t][a]['off'][0].mask[i]:
                    # rotate timestreams
                    I_off = meas_off_stream.I[i]
                    Q_off = meas_off_stream.Q[i]
                    
                    I_off_derot, Q_off_derot = derot_circle(I_off, Q_off, xc, yc, theta)

                    iq_circle.plot(I_off_derot, Q_off_derot, 'b.')

            iq_circle.axvline(0, color='k', lw=1)
            iq_circle.axhline(0, color='k', lw=1)
            
            iq_circle.set_xlabel(f'I[V]')
            iq_circle.set_ylabel(f'Q[V]')
            iq_circle.axis('equal')
            iq_circle.grid()

            self._create_diry(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/plots/{k}')
            fig_psd.savefig(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/plots/{k}/psd-{k}-{t}-{a}.png')
            close(fig_psd)

        if fit_psd:
            
            if psd_to_fit == 'on':
                selected_psd = psd_res_on
            
            elif psd_to_fit == 'on-off':
                selected_psd = psd_on_off

            # get resonator f0 and Qr
            try:

                flag_neigh = False
            
                if k in self.data['vna']:
                    if t in self.data['vna'][k]:
                        if a in self.data['vna'][k][t]:
                            if self.data['vna'][k][t][a][0].fit != {}:
                                f0 = self.data['vna'][k][t][a][0].fit['fr']
                                Q = self.data['vna'][k][t][a][0].fit['Qr']

                            else:
                                flag_neigh = True

                        else:
                            flag_neigh = True

                if flag_neigh:
                    
                    # get all the attens as numbers
                    attens_num = np.array([float(i[1:]) for i in self.data['vna'][k][t].keys()])
                    # get atten index
                    atten_idx = np.argmin( np.abs(attens_num - float(a[1:])) )

                    nw_atten = list(self.data['vna'][k][t].keys())[atten_idx]
                    printc(f'Attenuation not found, selecting the closest: {nw_atten} dB.', 'warn')

                    f0 = self.data['vna'][k][t][nw_atten][0].fit['fr']
                    Q = self.data['vna'][k][t][nw_atten][0].fit['Qr']

            except Exception as e:
                
                f0, Q = 1, 0
                printc(f'Ring-down time ignored.\n{e}', 'warn')

            # Purge the nan points
            purge_freq, purge_psd = [], []
            for i in range(len(selected_psd)):
                if not np.isnan(selected_psd[i]):
                    purge_psd.append(selected_psd[i])
                    purge_freq.append(freq_res_on[i])

            purge_freq = np.array(purge_freq)
            purge_psd = np.array(purge_psd)

            popt, fitted_psd = fit_res_psd(purge_freq, purge_psd, Q, f0, 
                                           save_path=f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/plots/{k}/fit-psd-{psd_to_fit}-{k}-{t}-{a}.png' )

            # save results
            self._create_diry(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}')
            try:
                np.save(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}/fitted-psd-{psd_to_fit}-{k}-{t}-{a}', fitted_psd)
            except Exception as e:
                printc(f'Fitted data not saved.\n{e}', 'fail')
            
            np.save(f'{self.work_dir}{self.project_name}/results/hmd/noise/psd/data/{k}/fit-params-{psd_to_fit}-{k}-{t}-{a}', popt)

            return freq_res_on, [psd_res_on, psd_res_off, psd_on_off], popt, fitted_psd

        else:

            return freq_res_on, [psd_res_on, psd_res_off, psd_on_off]


    def compute_single_psd(self, kid, temperature, attenuation, sample, tune='on', **kwargs):
        """
        Get the PSD of a given detector.
        Parameters
        ----------
        kid:            [int/str] detector id.
        temperature:    [int/str] temperature.
        attenuation:    [int/str attenuation.
        sample:         [int] sample number.
        tune:           [str] on/off resonance.
        ----------
        """

        # parse key format
        k = self._parse_single_item(kid, prefix='K', nzeros=3)                                          # get kid
        t = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)     # get temperature
        a = self._parse_single_item(attenuation, prefix='A', nzeros=3)                                  # get attenuation
        s = sample      # get sample

        available_sample = self._check_availability(k, t, a, sample=s, meas_type='hmd')

        if available_sample:
            
            if (self.data['hmd'][k][t][a][tune][s].df != None):
                # get df
                dfs = self.data['hmd'][k][t][a][tune][s].df

                # get fs
                fs = self.data['hmd'][k][t][a][tune][s].hdr['SAMPLERA']

                repeats = len(dfs)      # number of repeats
                for i, df in enumerate(dfs):
                    # get psd
                    frequency_psd, psd = get_psd(df, fs)
                    
                    if i == 0:
                        psd_accum = np.zeros_like(psd)
                    
                    psd_accum += psd

                self.data['hmd'][k][t][a][tune][s].psd = [frequency_psd, psd_accum/repeats]

            else:
                printc(f"Data sample {k}-{t}-{a}-{s} is empty, get df first.", "warn")

        else:
            printc(f"Data sample {k}-{t}-{a}-{s} not available.", "warn")


    def compute_single_df(self, kid, temperature, attenuation, sample, sweep_sample=0, 
                   ffs=False, tune='on', method='phase-inter', **kwargs):
        """
        Get the resonance frequency shift. 
        Parameters
        ----------
        kid:            [int/str] single detector id.
        temperature:    [int/str] single temperature.
        attenuation:    [int/str] single attenuation.
        sample:         [int] number of samples.
        sweep_sample:   [int] sweep data sample.
        ffs:            [bool] get fractional frequency shift?
        tune:           [str] 'on'/'off' resonance.
        method:         [str] 'phase-inter': phase interpolation.
                        [str] 'amp-tan': magic formula.
        **kwargs:       various parameters.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Frequency threshold to trim IQ circle and get its position and rotation.
        f0_thresh_circle_definition = kwargs.pop('f0_thresh_circle_definition', 8e4)
        # Frequency threshold to trim phase around the f0.
        f0_phase_thresh = kwargs.pop('f0_phase_thresh', 25e4)
        # Phase offset
        phase_offset = kwargs.pop('phase_offset', 0)
        # Smooth params phase
        smooth_params_phase = kwargs.pop('smooth_params_phase', [31, 3])
        # ----------------------------------------------

        assert method in ['amp-tan', 'phase-inter'], "Method not valid."

        # parse key format
        k = self._parse_single_item(kid, prefix='K', nzeros=3)                                          # get kid
        t = self._parse_single_item(temperature, prefix=self._temp_prefix, nzeros=self._temp_zeros)     # get temperature
        a = self._parse_single_item(attenuation, prefix='A', nzeros=3)                                  # get attenuation
        s = sample      # get sample

        available_sample = self._check_availability(k, t, a, sample=s, meas_type='hmd')

        if available_sample:

            # get I, Q
            I_timestream = self.data['hmd'][k][t][a][tune][s].I
            Q_timestream = self.data['hmd'][k][t][a][tune][s].Q

            nsamples = self.data['hmd'][k][t][a][tune][s].samples
            
            f0 = self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['F0']

            df_list = []

            # tangent amplitude
            if method == 'amp-tan':

                # get didf, dqdf
                didf = self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['DIDF']
                dqdf = self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['DQDF']

                # get I0, Q0
                I0 = self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['IF0']
                Q0 = self.data['hmd'][k][t][a]['sweep'][sweep_sample].hdr['QF0']

                for i in range(nsamples):
                    
                    # check mask
                    if self.data['hmd'][k][t][a][tune][s].mask[i]:

                        # get df
                        scale = 1
                        if ffs:
                            scale = f0 

                        df_list.append( df_from_magic(I_timestream[i], Q_timestream[i], didf, dqdf, I0, Q0)/scale )
                    
                    else:
                        printc(f"{k}-{t}-{a}-{s}, sample: {i} masked.", "warn")

                self.data['hmd'][k][t][a][tune][s].df = df_list

            # phase interpolation
            else:

                # get circle position and rotation angle.
                if not self.data['hmd'][k][t][a]['sweep'][sweep_sample].rotated:
                    self.data['hmd'][k][t][a]['sweep'][sweep_sample].get_circle_pos_and_rot(f0_thresh=f0_thresh_circle_definition)
                
                    # derotate circle
                    self.data['hmd'][k][t][a]['sweep'][sweep_sample].derotate_circle()

                # get phase model
                self.data['hmd'][k][t][a]['sweep'][sweep_sample].phase_model(smooth_params=smooth_params_phase,
                                                                            f0_phase_thresh=f0_phase_thresh, **kwargs)

                xc = self.data['hmd'][k][t][a]['sweep'][sweep_sample].xc.copy()
                yc = self.data['hmd'][k][t][a]['sweep'][sweep_sample].yc.copy()
                theta = self.data['hmd'][k][t][a]['sweep'][sweep_sample].theta.copy()
                
                phase_mdl = self.data['hmd'][k][t][a]['sweep'][sweep_sample].phase_mdl

                # need to derotate timestreams!
                for i in range(nsamples):

                    # check mask
                    if self.data['hmd'][k][t][a][tune][s].mask[i]:

                        # rotate timestreams
                        I_timestrem_derot, Q_timestream_derot = derot_circle(I_timestream[i], Q_timestream[i], xc, yc, theta)

                        if tune == 'off':
                            
                            # get gradient dQ/dI
                            dQ_dI = np.gradient(Q_timestream_derot, I_timestrem_derot)

                            # off resonance
                            try:
                                f0_off = self.data['hmd'][k][t][a][tune][s].hdr['SYNTHFRE']

                            except:
                                f0_off = self.data['hmd'][k][t][a][tune][s].hdr['F0']

                            meas_sweep = self.data['hmd'][k][t][a]['sweep'][sweep_sample]
                            
                            frequency = meas_sweep.frequency
                            f0_idx = np.nanargmin(np.abs(frequency-f0_off))

                            xc_p, yc_p = meas_sweep.s21.real[f0_idx], meas_sweep.s21.imag[f0_idx]
                            theta_p = np.arctan(dQ_dI[f0_idx])

                            # move to origin
                            I_ts_off_t1, Q_ts_off_t1 = derot_circle(I_timestrem_derot, Q_timestream_derot, xc_p, yc_p, theta_p)

                            # move to resonance
                            f0_on_idx = np.nanargmin(np.abs(frequency-f0))

                            xc_pp, yc_pp = meas_sweep.s21.real[f0_on_idx], meas_sweep.s21.imag[f0_on_idx]
                            theta_pp = np.arctan(dQ_dI[f0_on_idx])

                            if not np.isnan(theta_p) and not np.isnan(theta_pp):

                                # final correction
                                I_ts_off_t2, Q_ts_off_t2 = derot_circle(I_ts_off_t1, Q_ts_off_t1, 0, 0, theta_pp)

                                I_timestrem_derot, Q_timestream_derot = I_ts_off_t2 + xc_pp, Q_ts_off_t2 + yc_pp

                            else:
                                printc(f'Sample {i}: nan gradient!', 'fail')

                        s21_timestream_derot = I_timestrem_derot + 1j*Q_timestream_derot

                        # get df
                        scale = 1
                        if ffs:
                            scale = f0 

                        self.data['hmd'][k][t][a][tune][s].phase = get_phase(s21_timestream_derot)

                        df_list.append( df_from_phase(s21_timestream_derot, f0, phase_mdl, phase_offset)/scale )

                    else:
                        printc(f"{k}-{t}-{a}-{s}, sample: {i} masked.", "warn")

                self.data['hmd'][k][t][a][tune][s].df = df_list

            printc(f"df calculation for {k}-{t}-{a}-{s} done.", "ok")

        else:
            printc(f"Data sample {k}-{t}-{a}-{s} not available.", "warn")

        
    def _check_availability(self, kid, temperature, attenuation, sample=0, meas_type='vna', tune='on'):
        """
        Check if data is available.
        Parameters
        ----------
        kid:            [int/str/array] detector id.
        temperature:    [int/str/array] temperatures.
        attenuation:    [int/str/array] attenuations.
        samples:        [int] number of samples.
        meas_type:      [str] 'vna' or 'homodyne' measurements.
        tune:           [str] 'on'/'off' resonance.
        ----------
        """

        availability_status = False

        if kid in self.data[meas_type].keys():
            if temperature in self.data[meas_type][kid].keys():
                if attenuation in self.data[meas_type][kid][temperature].keys():
                    if meas_type == "hmd":
                        if sample in self.data[meas_type][kid][temperature][attenuation][tune].keys():
                            availability_status = True
                    elif sample in self.data[meas_type][kid][temperature][attenuation].keys():
                        availability_status = True
        
        return availability_status
    

    def _parse_items(self, items, prefix='K', nzeros=3):
        """
        Check kids format and convert to a list of valid elements.
        Parameters
        ----------
        items:      [int/float/array] items to parse.
        prefix:     [str] prefix to add at the beginning of the item.
        nzeros:     [int] number of zeros.
        ----------
        """

        items_list = []

        if isinstance(items, list) or isinstance(items, np.ndarray):
            
            for item in items:
                if isinstance(item, list):
                    subitem_list = []
                    for subitem in item:
                        str_subitem = self._parse_single_item(subitem, prefix=prefix, nzeros=nzeros)
                        subitem_list.append(str_subitem)

                    items_list.append(subitem_list)

                else:

                    str_item = self._parse_single_item(item, prefix=prefix, nzeros=nzeros)
                    items_list.append(str_item)               

        elif isinstance(items, int) or isinstance(items, float) or isinstance(items, str):
            items_list.append(self._parse_single_item(items, prefix=prefix, nzeros=nzeros))

        return items_list
    

    def _parse_single_item(self, item, prefix='K', nzeros=3):
        """
        Check a single element.
        Parameters
        ----------
        item:       [str/int/float] item to parse.
        prefix:     [str] prefix to add at the beginning of the item.
        nzeros:     [int] number of zeros.
        ----------
        """

        if isinstance(item, str):
            if item.isnumeric():
                item = f'{prefix}'+item.zfill(nzeros)
            elif item[1:].isnumeric():
                item = f'{prefix}'+item[1:].zfill(nzeros)
            
        elif isinstance(item, int) or isinstance(item, float) or isinstance(item, np.int64):

            if (isinstance(item, int) or isinstance(item, np.int64)) and prefix == 'A':
                item = float(item)

            item = f'{prefix}'+str(item).zfill(nzeros)

        return item
    
    def _fit_res_join(self, kid, temp, atten, f, s21, n, tau):
        """
        Fit resonator job.
        Parameters
        ----------
        kid:            [str]
        temp:           [str]
        atten:          [str]
        frequency:      [array] frequency array [Hz].
        s21:            [array] s21 array.
        n:              [float] selected data fraction.
        tau:            [float] time delay [s].
        ----------
        """

        fit_res = fit_single_resonator(f, s21, n=n, tau=tau)
        printc(f'{kid} - {temp} - {atten}', 'ok')

        fitRes[kid+','+temp+','+atten] = fit_res


