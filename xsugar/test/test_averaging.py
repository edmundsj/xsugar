"""
Tests data reading and writing operation, along with condition generation
"""
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment
from ast import literal_eval
from itertools import zip_longest
from spectralpy import power_spectrum
from sciparse import assertDataDictEqual

@pytest.fixture
def exp(path_data):
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    exp = Experiment(name='TEST1', kind='test',
                     frequency=frequency,
                     wavelengths=wavelengths,
                     temperatures=temperatures)
    yield exp
    rmtree(path_data['data_base_path'], ignore_errors=True)
    rmtree(path_data['figures_base_path'], ignore_errors=True)
    rmtree(path_data['designs_base_path'], ignore_errors=True)

def test_average_data_scalar(exp, convert_name):
    fudge_data_1 = 2.0
    fudge_data_2 = 3.0
    averaged_data = 2.5
    condition_1 = {'frequency': 8500,
                   'wavelength': 1,
                   'replicate': 0}
    condition_2 = {'frequency': 8500,
                   'wavelength': 1,
                   'replicate': 1}
    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}
    group_name = convert_name('TEST1~wavelength-1')

    averaged_data_desired = {group_name: averaged_data}
    averaged_data_actual = exp.average_data(data_dict, average_along='replicate')
    assertDataDictEqual(averaged_data_actual, averaged_data_desired)

def test_average_data_pandas(exp, convert_name):
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 1.2, 1.4]})
    averaged_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.75, 0.9, 1.05]})
    condition_1 = {'frequency': 8500,
                   'wavelength': 1,
                   'replicate': 0}
    condition_2 = {'frequency': 8500,
                   'wavelength': 1,
                   'replicate': 1}
    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    group_name = convert_name('TEST1~wavelength-1')
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}

    averaged_data_desired = {group_name: averaged_data}
    averaged_data_actual_first = exp.average_data(data_dict,
            average_along='replicate', averaging_type='first')
    averaged_data_actual_last = exp.average_data(data_dict,
            average_along='replicate', averaging_type='last')
    assertDataDictEqual(averaged_data_actual_first, averaged_data_desired)
    assertDataDictEqual(averaged_data_actual_last, averaged_data_desired)

def test_average_data_unsupported(exp):
    with pytest.raises(ValueError):
        averaged_data_actual = exp.average_data(
                average_along='replicate', averaging_type='YOLO')

def testExtractDerivedQuantityMean(exp, convert_name):
    """
    Attempst to extract the mean from a set of data
    """
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 1.2, 1.4]})
    mean_data_1 = np.float64(0.6)
    mean_data_2 = np.float64(1.2)
    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    desired_quantities = {name_1: mean_data_1, name_2: mean_data_2}
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}

    def getPhotocurrentMean(pandas_dict, cond):
        return np.mean(pandas_dict['Photocurrent (nA)'].values)

    actual_quantities = exp.derived_quantity(
        data_dict=data_dict, quantity_func=getPhotocurrentMean,
        average_along=None)

    assertDataDictEqual(actual_quantities, desired_quantities)

def testExtractDerivedQuantityPSD(exp, convert_name):
    """
    Attempts to extract the PSD from a set of data
    """
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 1.2, 1.4]})
    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}

    def powerSpectrum(data, cond):
        return power_spectrum(data)
    actual_psd_data = exp.derived_quantity(
        data_dict=data_dict, quantity_func=powerSpectrum)
    desired_psd_data = {name_1: power_spectrum(fudge_data_1),
                        name_2: power_spectrum(fudge_data_2)}
    assertDataDictEqual(actual_psd_data, desired_psd_data)

def testExtractDerivedQuantityPSDAverage(exp, convert_name):
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 1.2, 1.4]})
    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}
    data_psd_1 = power_spectrum(fudge_data_1)
    data_psd_2 = power_spectrum(fudge_data_2)
    data_psd_1.iloc[:, 1:] += data_psd_2.iloc[:,1:]
    data_psd_1.iloc[:, 1:] /= 2

    def powerSpectrum(data, cond):
        return power_spectrum(data)

    desired_psd = data_psd_1
    desired_data_dict = {convert_name('TEST1~wavelength-1'): desired_psd}
    actual_data_dict = exp.derived_quantity(
        data_dict=data_dict, quantity_func=powerSpectrum,
        average_along='replicate')
    assertDataDictEqual(actual_data_dict, desired_data_dict)
