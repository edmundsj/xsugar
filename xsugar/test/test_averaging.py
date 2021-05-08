"""
@pytest.fixture
def data_2var_summed():
    summed_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [6, 10, 14]})
    yield summed_data
Tests data reading and writing operation, along with condition generation
"""
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment, ureg
from ast import literal_eval
from itertools import zip_longest
from spectralpy import power_spectrum
from sciparse import assertDataDictEqual

@pytest.fixture
def data_2x2_scalar():
    wavelengths = [1, 2]
    temperatures = [25, 50]
    index = pd.MultiIndex.from_product([wavelengths, temperatures],
            names=['wavelength', 'temperature'])
    data = pd.DataFrame(index=index,data={'value': [1, 2, 3, 4]})
    yield data

@pytest.fixture
def data_2x2_averaged_scalar():
    wavelengths = [1, 2]
    index = pd.Index(wavelengths ,name='wavelength')
    data = pd.DataFrame(index=index,data={'value': [1.5, 3.5]})
    yield data

@pytest.fixture
def data_2x2_pandas():
    wavelengths = [1, 1, 2, 2]
    temperatures = [25, 50, 25, 50]
    index = pd.MultiIndex.from_arrays([wavelengths, temperatures],
            names=['wavelength', 'temperature'])
    inner_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 2, 3]})
    inner_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [2, 3, 4]})
    data = pd.DataFrame(index=index,
            data={'series':
                [inner_data_1, inner_data_2,
                inner_data_1, inner_data_2]})
    yield data

@pytest.fixture
def data_2x2_pandas_averaged():
    wavelengths = [1, 2]
    index = pd.Index(wavelengths, name='wavelength')
    averaged_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1.5, 2.5, 3.5]})
    data = pd.DataFrame(index=index,
            data={'series': [averaged_data, averaged_data]})
    yield data

@pytest.fixture
def data_2x2_pandas_summed():
    wavelengths = [1, 2]
    index = pd.Index(wavelengths, name='wavelength')
    summed_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [3, 5, 7]})
    data = pd.DataFrame(index=index,
            data={'series': [summed_data, summed_data]})
    yield data

@pytest.fixture
def exp(path_data):
    frequency = 8500
    exp = Experiment(name='TEST1', kind='test',
                     frequency=frequency)
    yield exp
    rmtree(path_data['data_base_path'], ignore_errors=True)
    rmtree(path_data['figures_base_path'], ignore_errors=True)
    rmtree(path_data['designs_base_path'], ignore_errors=True)

def test_average_data_scalar(exp,
        data_2x2_scalar, data_2x2_averaged_scalar):
    averaged_data_desired = data_2x2_averaged_scalar
    averaged_data_actual = exp.mean(
            data_2x2_scalar, along='temperature')
    assert_frame_equal(averaged_data_actual, averaged_data_desired)

def test_sum_data_scalar(exp,
        data_2x2_scalar, data_2x2_averaged_scalar):
    averaged_data_desired = data_2x2_averaged_scalar * 2
    averaged_data_actual = exp.sum(
            data_2x2_scalar, along='temperature') * 1.0
    assert_frame_equal(averaged_data_actual, averaged_data_desired)

def test_average_data_pandas(exp,
        data_2x2_pandas, data_2x2_pandas_averaged):

    averaged_data_desired = data_2x2_pandas_averaged
    averaged_data_actual = exp.mean(
            data=data_2x2_pandas,
            along='temperature')
    # Need a more elegant way of testing nested frame equality.
    assert_index_equal(averaged_data_actual.index,
            averaged_data_desired.index)
    assert_frame_equal(averaged_data_actual.iloc[0, 0],
            averaged_data_desired.iloc[0, 0])
    assert_frame_equal(averaged_data_actual.iloc[1, 0],
            averaged_data_desired.iloc[1, 0])

def test_sum_data_pandas(exp,
        data_2x2_pandas, data_2x2_pandas_summed):

    summed_data_desired = data_2x2_pandas_summed
    summed_data_actual = exp.sum(
            data=data_2x2_pandas, along='temperature')

    assert_index_equal(summed_data_actual.index,
            summed_data_desired.index)
    assert_frame_equal(summed_data_actual.iloc[0, 0],
            summed_data_desired.iloc[0, 0])
    assert_frame_equal(summed_data_actual.iloc[1, 0],
            summed_data_desired.iloc[1, 0])

def test_derived_quantity_scalar(exp, data_2x2_scalar):
    def multiply_func(data, cond):
        return data * 2

    actual_quantities = exp.derived_quantity(quantity_func=multiply_func,
            data=data_2x2_scalar)
    desired_quantities = data_2x2_scalar * 2
    assert_frame_equal(actual_quantities, desired_quantities)

def test_derived_quantity_pandas(exp, data_2x2_pandas):
    """
    Attempst to extract the mean from a set of data
    """
    def getPhotocurrentMean(pandas_frame, cond):
        return np.mean(pandas_frame['Photocurrent (nA)'].values)

    actual_quantities = exp.derived_quantity(
        data=data_2x2_pandas, quantity_func=getPhotocurrentMean,
        average_along=None)

    breakpoint()
    assertDataDictEqual(actual_quantities, desired_quantities)

def test_derived_quantity_sum(exp, convert_name):
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [1, 1.2, 1.4]})

    name_1 = convert_name('TEST1~wavelength-1~replicate-0')
    name_2 = convert_name('TEST1~wavelength-1~replicate-1')
    group_name = convert_name('TEST1~wavelength-1')
    desired_quantities = {group_name: 11.400000000000002*ureg.pC}
    data_dict = {name_1: fudge_data_1, name_2: fudge_data_2}

    def charge(data, cond):
        charges = data['Photocurrent (nA)'].values* \
            data['Time (ms)'].values*ureg.nA*ureg.ms
        total_charge = charges.sum().to(ureg.pC)
        return total_charge

    actual_quantities = exp.derived_quantity(
        data_dict=data_dict, quantity_func=charge,
        sum_along='replicate')

    assertDataDictEqual(actual_quantities, desired_quantities)

def test_derived_quantity_psd(exp, convert_name):
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

def test_derived_quantity_psd_average(exp, convert_name):
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

# TODO: Derived quantities where we have arrays instead of DataFrames or scalars.
