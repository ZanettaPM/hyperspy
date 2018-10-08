# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import warnings
import logging
import math
import copy
import xlsxwriter

import traits.api as t
import numpy as np
from scipy import constants
from scipy import pi

from hyperspy.signal import BaseSetMetadataItems
from hyperspy import utils
from hyperspy._signals.eds import (EDSSpectrum, LazyEDSSpectrum)
from hyperspy.defaults_parser import preferences
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.ui_registry import add_gui_method, DISPLAY_DT, TOOLKIT_DT
from hyperspy.misc.eds.kfactors import get_kfactors
from hyperspy.misc.eds.correct_result_2D import correct_result_2D
from hyperspy.misc.material import _mass_absorption_mixture as mass_absorption_mixture
from hyperspy.misc.material import weight_to_atomic

_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="microscope_parameters_EDS_TEM")
class EDSTEMParametersUI(BaseSetMetadataItems):
    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    real_time = t.Float(t.Undefined,
                        label='Real time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    probe_area = t.Float(t.Undefined,
                         label='Beam/probe area (nm\xB2)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')
    energy_resolution_MnKa = t.Float(t.Undefined,
                                     label='Energy resolution MnKa (eV)')
    beam_current = t.Float(t.Undefined,
                           label='Beam current (nA)')
    mapping = {
        'Acquisition_instrument.TEM.beam_energy': 'beam_energy',
        'Acquisition_instrument.TEM.Stage.tilt_alpha': 'tilt_stage',
        'Acquisition_instrument.TEM.Detector.EDS.live_time': 'live_time',
        'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
        'azimuth_angle',
        'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
        'elevation_angle',
        'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
        'energy_resolution_MnKa',
        'Acquisition_instrument.TEM.beam_current':
        'beam_current',
        'Acquisition_instrument.TEM.probe_area':
        'probe_area',
        'Acquisition_instrument.TEM.Detector.EDS.real_time':
        'real_time', }


class EDSTEM_mixin:

    _signal_type = "EDS_TEM"

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.TEM.Detector.EDS' not in self.metadata:
            if 'Acquisition_instrument.SEM.Detector.EDS' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self._set_default_param()

    def _set_default_param(self):
        """Set to value to default (defined in preferences)
        """

        mp = self.metadata
        mp.Signal.signal_type = "EDS_TEM"

        mp = self.metadata
        if "Acquisition_instrument.TEM.Stage.tilt_alpha" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.TEM.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa"\
                not in mp:
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS." +
                        "energy_resolution_MnKa",
                        preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                preferences.EDS.eds_detector_azimuth)

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  live_time=None,
                                  tilt_stage=None,
                                  azimuth_angle=None,
                                  elevation_angle=None,
                                  energy_resolution_MnKa=None,
                                  beam_current=None,
                                  probe_area=None,
                                  real_time=None,
                                  display=True,
                                  toolkit=None):
        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa, beam_current,
                probe_area, real_time]) == {None}:
            tem_par = EDSTEMParametersUI(self)
            return tem_par.gui(display=display, toolkit=toolkit)
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy ", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                elevation_angle)
        if energy_resolution_MnKa is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS." +
                "energy_resolution_MnKa",
                energy_resolution_MnKa)
        if beam_current is not None:
            md.set_item(
                "Acquisition_instrument.TEM.beam_current",
                beam_current)
        if probe_area is not None:
            md.set_item(
                "Acquisition_instrument.TEM.probe_area",
                probe_area)
        if real_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.real_time",
                real_time)

    set_microscope_parameters.__doc__ = \
        """
        Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        live_time : float
            In seconds
        tilt_stage : float
            In degree
        azimuth_angle : float
            In degree
        elevation_angle : float
            In degree
        energy_resolution_MnKa : float
            In eV
        beam_current: float
            In nA
        probe_area: float
            In nm\xB2
        real_time: float
            In seconds
        {}
        {}

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        >>> s.set_microscope_parameters(energy_resolution_MnKa=135.)
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        133.312296
        135.0

        """.format(DISPLAY_DT, TOOLKIT_DT)

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in metadata. Raise in interactive mode
         an UI item to fill or change the values"""
        must_exist = (
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EDS.live_time',)

        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            _logger.info("Missing parameters {}".format(missing_parameters))
            return True
        else:
            return False

    def get_calibration_from(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA
        software

        Parameters
        ----------
        ref : signal
            The reference contains the calibration in its
            metadata
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an
            average live time.

        Examples
        --------
        >>> ref = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s = hs.signals.EDSTEMSpectrum(
        >>>     hs.datasets.example_signals.EDS_TEM_Spectrum().data)
        >>> print(s.axes_manager[0].scale)
        >>> s.get_calibration_from(ref)
        >>> print(s.axes_manager[0].scale)
        1.0
        0.020028

        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # Setup metadata
        if 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        elif 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        else:
            raise ValueError("The reference has no metadata." +
                             "Acquisition_instrument.TEM" +
                             "\n or metadata.Acquisition_instrument.SEM ")

        mp = self.metadata
        mp.Acquisition_instrument.TEM = mp_ref.deepcopy()
        if mp_ref.has_item("Detector.EDS.live_time"):
            mp.Acquisition_instrument.TEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def quantification(self,
                       intensities,
                       method,
                       factors='auto',
                       composition_units='atomic',
                       navigation_mask=1.0,
                       closing=True,
                       plot_result=False,
                       **kwargs):
        """
        Quantification using Cliff-Lorimer, the zeta-factor method, or
        ionization cross sections.

        Parameters
        ----------
        intensities: list of signal
            the intensitiy for each X-ray lines.
        method: 'CL' or 'zeta' or 'cross_section'
            Set the quantification method: Cliff-Lorimer, zeta-factor, or
            ionization cross sections.
        factors: list of float
            The list of kfactors, zeta-factors or cross sections in same order
            as intensities. Note that intensities provided by Hyperspy are
            sorted by the alphabetical order of the X-ray lines.
            eg. factors =[0.982, 1.32, 1.60] for ['Al_Ka', 'Cr_Ka', 'Ni_Ka'].
        composition_units: 'weight' or 'atomic'
            The quantification returns the composition in atomic percent by
            default, but can also return weight percent if specified.
        navigation_mask : None or float or signal
            The navigation locations marked as True are not used in the
            quantification. If int is given the vacuum_mask method is used to
            generate a mask with the int value as threhsold.
            Else provides a signal with the navigation shape.
        closing: bool
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        plot_result : bool
            If True, plot the calculated composition. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments are passed to plot.

        Returns
        ------
        A list of quantified elemental maps (signal) giving the composition of
        the sample in weight or atomic percent.

        If the method is 'zeta' this function also returns the mass thickness
        profile for the data.

        If the method is 'cross_section' this function also returns the atom
        counts for each element.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> kfactors = [1.450226, 5.075602] #For Fe Ka and Pt La
        >>> bw = s.estimate_background_windows(line_width=[5.0, 2.0])
        >>> s.plot(background_windows=bw)
        >>> intensities = s.get_lines_intensity(background_windows=bw)
        >>> res = s.quantification(intensities, kfactors, plot_result=True,
        >>>                        composition_units='atomic')
        Fe (Fe_Ka): Composition = 15.41 atomic percent
        Pt (Pt_La): Composition = 84.59 atomic percent

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        elif navigation_mask is not None:
            navigation_mask = navigation_mask.data
        xray_lines = self.metadata.Sample.xray_lines
        composition = utils.stack(intensities, lazy=False)
        if method == 'CL':
            composition.data = utils_eds.quantification_cliff_lorimer(
                composition.data, kfactors=factors,
                mask=navigation_mask) * 100.
        elif method == 'zeta':
            results = utils_eds.quantification_zeta_factor(
                composition.data, zfactors=factors,
                dose=self._get_dose(method, **kwargs))
            composition.data = results[0] * 100.
            mass_thickness = intensities[0].deepcopy()
            mass_thickness.data = results[1]
            mass_thickness.metadata.General.title = 'Mass thickness'
        elif method == 'cross_section':
            results = utils_eds.quantification_cross_section(
                composition.data,
                cross_sections=factors,
                dose=self._get_dose(method, **kwargs))
            composition.data = results[0] * 100
            number_of_atoms = composition._deepcopy_with_new_data(results[1])
            number_of_atoms = number_of_atoms.split()
        else:
            raise ValueError('Please specify method for quantification,'
                             'as \'CL\', \'zeta\' or \'cross_section\'')
        composition = composition.split()
        if composition_units == 'atomic':
            if method != 'cross_section':
                composition = utils.material.weight_to_atomic(composition)
        else:
            if method == 'cross_section':
                composition = utils.material.atomic_to_weight(composition)
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            composition[i].metadata.General.title = composition_units + \
                ' percent of ' + element
            composition[i].metadata.set_item("Sample.elements", ([element]))
            composition[i].metadata.set_item(
                "Sample.xray_lines", ([xray_line]))
            if plot_result and \
                    composition[i].axes_manager.navigation_size == 1:
                print("%s (%s): Composition = %.2f %s percent"
                      % (element, xray_line, composition[i].data,
                         composition_units))
        if method == 'cross_section':
            for i, xray_line in enumerate(xray_lines):
                element, line = utils_eds._get_element_and_line(xray_line)
                number_of_atoms[i].metadata.General.title = \
                    'atom counts of ' + element
                number_of_atoms[i].metadata.set_item("Sample.elements",
                                                     ([element]))
                number_of_atoms[i].metadata.set_item(
                    "Sample.xray_lines", ([xray_line]))
        if plot_result and composition[i].axes_manager.navigation_size != 1:
            utils.plot.plot_signals(composition, **kwargs)
        if method == 'zeta':
            self.metadata.set_item("Sample.mass_thickness", mass_thickness)
            return composition, mass_thickness
        elif method == 'cross_section':
            return composition, number_of_atoms
        elif method == 'CL':
            return composition
        else:
            raise ValueError('Please specify method for quantification, as \
            ''CL\', \'zeta\' or \'cross_section\'')

    def vacuum_mask(self, threshold=1.0, closing=True, opening=False):
        """
        Generate mask of the vacuum region

        Parameters
        ----------
        threshold: float
            For a given pixel, maximum value in the energy axis below which the
            pixel is considered as vacuum.
        closing: bool
            If true, applied a morphologic closing to the mask
        opnening: bool
            If true, applied a morphologic opening to the mask

        Examples
        --------
        >>> # Simulate a spectrum image with vacuum region
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s_vac = hs.signals.BaseSignal(
                np.ones_like(s.data, dtype=float))*0.005
        >>> s_vac.add_poissonian_noise()
        >>> si = hs.stack([s]*3 + [s_vac])
        >>> si.vacuum_mask().data
        array([False, False, False,  True], dtype=bool)

        Return
        ------
        mask: signal
            The mask of the region
        """
        from scipy.ndimage.morphology import binary_dilation, binary_erosion
        mask = (self.max(-1) <= threshold)
        if closing:
            mask.data = binary_dilation(mask.data, border_value=0)
            mask.data = binary_erosion(mask.data, border_value=1)
        if opening:
            mask.data = binary_erosion(mask.data, border_value=1)
            mask.data = binary_dilation(mask.data, border_value=0)
        return mask

    
    def correction(self, elts, Quant, result_int, result_mod, alpha, mt):
        
        """    F_Si = 0.005
        F_S = 0.01
        F_C = 0.00015
        F_Ca = 0.0025
        F_O = 0.006
        F_Fe = 0.045"""
        
        Ac = np.zeros((len(Quant[0].data), len(Quant[0].data[0]), len(Quant)), 'float')
        wt = np.zeros((len(Quant[0].data), len(Quant[0].data[0]), len(Quant)), 'float')
        
        for i in range (0, len(Quant[0].data)):
            for j in range (0, len(Quant[0].data[0])):
                if navigation_mask.data[i][j]== False:
                    for k in range(0, len(self.metadata.Sample.xray_lines)):
                        wt[i][j][k]=Quant[k].data[i][j]
            
        print('initialization of Ac and wt is OK')

        """wt_Si= np.zeros((len(Quant[0].data), len(Quant[0].data[0])), float)
        wt_S= np.zeros((len(Quant[0].data), len(Quant[0].data[0])), float)
        wt_C= np.zeros((len(Quant[0].data), len(Quant[0].data[0])), float)
        wt_Ca= np.zeros((len(Quant[0].data), len(Quant[0].data[0])), float)
        wt_O =np.zeros((len(Quant[0].data), len(Quant[0].data[0])), float)

        for k in range(0, len(Quant)):                  
            if 'Si_Ka' in Quant[k].metadata.Sample.xray_lines: 
                wt_Si = Quant[k].data
            if 'S_Ka' in Quant[k].metadata.Sample.xray_lines: 
                wt_S = Quant[k].data
            if 'C_Ka' in Quant[k].metadata.Sample.xray_lines: 
                wt_C = Quant[k].data
            if 'Ca_Ka' in Quant[k].metadata.Sample.xray_lines: 
                wt_Ca = Quant[k].data
            if 'O_Ka' in Quant[k].metadata.Sample.xray_lines: 
                wt_O = Quant[k].data"""                       

        for i in range(0, len(Quant[0].data)):
            for j in range (0, len(Quant[0].data[0])):
                if navigation_mask.data[i][j]== False:
                    for k in range (0, len(self.metadata.Sample.xray_lines)):
                        Ac[i][j][k] = hs.material.mass_absorption_mixture(wt[i][j], elts, energies = self.metadata.Sample.xray_lines[k])
                        """if 'C_Ka' in Quant[k].metadata.Sample.xray_lines: 
                            Ac[i][j][k] = (1+14*F_Si*wt_Si[i][j]+F_S*wt_S[i][j]+F_C*wt[i][j][k])*Ac[i][j][k]
                        if 'Ca_La' in Quant[k].metadata.Sample.xray_lines: 
                            Ac[i][j][k] = (1+F_Si*wt_Si[i][j]+F_S*wt_S[i][j]+0.5*F_C*wt_C[i][j]+F_Ca*wt[i][j][k])*Ac[i][j][k]
                        if 'O_Ka' in Quant[k].metadata.Sample.xray_lines: 
                            Ac[i][j][k] = (1+F_Si*wt_Si[i][j]+F_S*wt_S[i][j]+F_C*wt_C[i][j]+0.1*F_Ca*wt_Ca[i][j]+F_O*wt[i][j][k])*Ac[i][j][k]
                        if 'Fe_La' in Quant[k].metadata.Sample.xray_lines: 
                            Ac[i][j][k] = (1+F_Si*wt_Si[i][j]+F_S*wt_S[i][j]+F_C*wt_C[i][j]+0.01*F_Ca*wt_Ca[i][j]+F_O*wt_O[i][j]+F_Fe*wt[i][j][k])*Ac[i][j][k]
    """
        print('AC calculation OK')
        
        #Calculate the corrected intensities thanks to the abs. correction factors           
        for i in range (0, len(Quant[0].data)):
            for j in range (0, len(Quant[0].data[0])):
                if navigation_mask.data[i][j]== False:
                    for k in range (0, len(self.metadata.Sample.xray_lines)):
                        result_mod[k].data[i][j] = result_int[k].data[i][j]*Ac[i][j][k]/(1-np.exp(-(Ac[i][j][k])*mt[i][j]/np.sin(alpha)))
        return result_mod

    def absorption_correction_2D(self,result, kfactors='From_database', d = 3, t = 150, tilt_stage = 0, navigation_mask = 1.0):

        result2=correct_result_2D(result)

        if kfactors == 'From_database' :
            kfactors=get_kfactors(result2)
            print('kfactors',kfactors)
        else:
            kfactors = kfactors
            print('kfactors',kfactors)
       
        self.metadata.Acquisition_instrument.TEM.tilt_stage = tilt_stage
        alpha = (self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle + 
             self.metadata.Acquisition_instrument.TEM.tilt_stage)*pi/180 
            
        if t == 0:
            t = 1
       
        mt = np.ones((len(result2[0].data), len(result2[0].data[0])), float)
        for i in range (0, len(result2[0].data)):
            for j in range (0, len(result2[0].data[0])):
                if navigation_mask.data[i][j]== False:
                    mt[i][j]=(d*(t*10**-7))   
        
        elts = []
        for i in range(0, len(self.metadata.Sample.xray_lines)):
            elts.append(result2[i].metadata.Sample.elements)   
        
        dif = np.ones((len(self.metadata.Sample.xray_lines), len(result2[0].data), len(result2[0].data[0])), float)
        
        print('elts', elts)
        
        result_int = copy.deepcopy(result2) # since result is manipulated many times, better copy it deeply before.
        result_mod = copy.deepcopy(result2) 

        Quant = self.quantification(method="CL", intensities=result_int, factors=kfactors, composition_units='weight', navigation_mask=navigation_mask, plot_result=False)   
        Quant2 = Quant
        
        while (abs(dif) > 0.005).any():
            Quant = Quant2
            
            #Calculation of intensities corrected for absorption
            result_mod = self.correction (elts, Quant, result_int, result_mod, alpha, mt)

            #New quantification using corrected intensities
            Quant2 = self.quantification(method="CL", intensities=result_mod, factors=kfactors, composition_units='weight', navigation_mask=navigation_mask, plot_result=False)          
      
            #Compares the relative difference between previous and last quantification (convergence test value)
            dif = np.zeros((len(self.metadata.Sample.xray_lines), len(result2[0].data), len(result2[0].data[0])), float)

            for i in range (0,len(Quant)):
                dif[i]= abs(Quant[i].data-Quant2[i].data)/Quant[i].data
                dif[i] = dif[i].flatten()
            for i in range (0, len(dif)):
                if math.isnan(dif[i]):dif[i] = 0
            dif = np.round(dif, decimals = 3)

        Quant3 = weight_to_atomic(Quant2, elements='auto')
        return Quant3, dif, mt


    def decomposition(self,
                      normalize_poissonian_noise=True,
                      navigation_mask=1.0,
                      closing=True,
                      *args,
                      **kwargs):
        """
        Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        navigation_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decomposition. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> si = hs.stack([s]*3)
        >>> si.change_dtype(float)
        >>> si.decomposition()

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        super().decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask, *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

    def create_model(self, auto_background=True, auto_add_lines=True,
                     *args, **kwargs):
        """Create a model for the current TEM EDS data.

        Parameters
        ----------
        auto_background : boolean, default True
            If True, adds automatically a polynomial order 6 to the model,
            using the edsmodel.add_polynomial_background method.
        auto_add_lines : boolean, default True
            If True, automatically add Gaussians for all X-rays generated in
            the energy range by an element using the edsmodel.add_family_lines
            method.
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------

        model : `EDSTEMModel` instance.

        """
        from hyperspy.models.edstemmodel import EDSTEMModel
        model = EDSTEMModel(self,
                            auto_background=auto_background,
                            auto_add_lines=auto_add_lines,
                            *args, **kwargs)
        return model

    def _get_dose(self, method, beam_current='auto', live_time='auto',
                  probe_area='auto', **kwargs):
        """
        Calculates the total electron dose for the zeta-factor or cross section
        methods of quantification.

        Input given by i*t*N, i the current, t the
        acquisition time, and N the number of electron by unit electric charge.

        Parameters
        ----------
        method : 'zeta' or 'cross_section'
            If 'zeta', the dose is given by i*t*N
            If 'cross section', the dose is given by i*t*N/A
            where i is the beam current, t is the acquistion time,
            N is the number of electrons per unit charge (1/e) and
            A is the illuminated beam area or pixel area.
        beam_current: float
            Probe current in nA
        live_time: float
            Acquisiton time in s, compensated for the dead time of the detector.
        probe_area: float
            The illumination area of the electron beam in nm\xB2.
            If not set the value is extracted from the scale axes_manager.
            Therefore we assume the probe is oversampling such that
            the illumination area can be approximated to the pixel area of the
            spectrum image.

        Returns
        --------
        Dose in electrons (zeta factor) or electrons per nm\xB2 (cross_section)

        See also
        --------
        set_microscope_parameters
        """

        parameters = self.metadata.Acquisition_instrument.TEM

        if beam_current is 'auto':
            if 'beam_current' not in parameters:
                raise Exception('Electron dose could not be calculated as\
                     beam_current is not set.'
                                'The beam current can be set by calling \
                                set_microscope_parameters()')
            else:
                beam_current = parameters.beam_current

        if live_time == 'auto':
            live_time = parameters.Detector.EDS.live_time
            if 'live_time' not in parameters.Detector.EDS:
                raise Exception('Electron dose could not be calculated as \
                live_time is not set. '
                                'The beam_current can be set by calling \
                                set_microscope_parameters()')
            elif live_time == 1:
                warnings.warn('Please note that your real time is set to '
                              'the default value of 0.5 s. If this is not \
                              correct, you should change it using '
                              'set_microscope_parameters() and run \
                              quantification again.')

        if method == 'cross_section':
            if probe_area == 'auto':
                if probe_area in parameters:
                    area = parameters.TEM.probe_area
                else:
                    pixel1 = self.axes_manager[0].scale
                    pixel2 = self.axes_manager[1].scale
                    if pixel1 == 1 or pixel2 == 1:
                        warnings.warn('Please note your probe_area is set to'
                                      'the default value of 1 nm\xB2. The \
                                      function will still run. However if'
                                      '1 nm\xB2 is not correct, please read the \
                                      user documentations for how to set this \
                                      properly.')
                    area = pixel1 * pixel2
            return (live_time * beam_current * 1e-9) / (constants.e * area)
            # 1e-9 is included here because the beam_current is in nA.
        elif method == 'zeta':
            return live_time * beam_current * 1e-9 / constants.e
        else:
            raise Exception('Method need to be \'zeta\' or \'cross_section\'.')


    def data_output_2D (self, Quant, result, mt_List, H2O_List=None, density_or_thickness=2.7, name = 'data.xlsx'):    #H2O_List, mt_List, density_or_thickness, Dev

        result2=correct_result_2D(result)
        
        workbook = xlsxwriter.Workbook(name)
        worksheet_table = workbook.add_worksheet(name = 'Quantifed data')
        worksheet_error = workbook.add_worksheet(name = 'Errors')
        
        Q = np.ndarray((len(Quant), len((Quant[0].data).flatten())))
        for i in range (len (Quant)):
            Q[i] = (Quant[i].data).flatten()


        R = np.ndarray((len(result), len((result[0].data).flatten())))
        for i in range (len (result)):
            R[i] = (result[i].data).flatten()
        
        mt = mt_List.flatten()
        
        if H2O_List is not None:
            H2O = H2O_List.flatten()

        d_or_t = np.ones(shape=mt.shape, dtype=float)
        for i in range (0, len(d_or_t)):
            d_or_t[i]= density_or_thickness


        row = []
        for i in range (0, len(Quant)):
            row.append(0)
            row [i] = str(Quant[i].metadata.Sample.xray_lines)
        worksheet_table.write_row ('B1', row)
        worksheet_error.write_row ('B1', row)
        if H2O_List is not None:
            worksheet_table.write (0, len(row)+1, 'H2O')
        worksheet_table.write(0, len(row)+2, 'Deduced t or d')
        worksheet_table.write(0, len(row)+3, 'Chosen t or d')
        #worksheet_table.write(0, len(row)+4, 'Auto Abs . Cor. deviation')
        
        for i in range(len(Q)):
            k=0
            for j in range (len(Q[0])):
                if Q[:,j].any() != 0:
                    worksheet_table.write(k+1,i+1, Q[i][j]) 
                    if H2O_List is not None:
                        worksheet_table.write (i+1, len(row)+1, H2O[i])
                worksheet_table.write (i+1, len(row)+2, mt[i]/(d_or_t[i]*10**-7))
                worksheet_table.write (i+1, len(row)+3, d_or_t[i])
                #worksheet_table.write (i+1, len(row)+4, Dev[i])
                k=k+1
        for i in range(len(R)):
            k=0
            for j in range (len(R[0])):
                if Q[:,j].any() != 0 and R[i][j]!=0:
                    worksheet_error.write(k+1,i+1, (Q[i][j]*R[i][j]**0.5)/(R[i][j]))
                    k=k+1
            workbook.close()


class EDSTEMSpectrum(EDSTEM_mixin, EDSSpectrum):
    pass


class LazyEDSTEMSpectrum(EDSTEMSpectrum, LazyEDSSpectrum):
    pass
