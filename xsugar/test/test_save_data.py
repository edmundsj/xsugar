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
from sciparse import dict_to_string, assertDataDictEqual
from ast import literal_eval
from itertools import zip_longest
from pathlib import Path

def testExtractConstants(exp, exp_data):
    """
    Tests that our experiment properly extracts metadata from a list of
    arbitrary keyword arguments.
    """
    actual_constants = exp.constants
    desired_constants = {'frequency': exp_data['frequency']}
    assert_equal(actual_constants , desired_constants)

def test_save_raw_results_filename(exp, exp_data):
    """
    Tests that we save the raw results in the proper directory and with
    the proper metadata and with the proper name.
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    raw_data = pd.DataFrame({'wavelength': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    exp.saveRawResults(raw_data, cond)
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
                       'temperature' + ns + '25.csv'
    file_found = os.path.isfile(exp_data['data_full_path'] + filename_desired)
    assert_equal(file_found, True)

def test_save_raw_results_metadata(exp, exp_data):
    """
    Tests that we save the raw results in the proper directory and with
    the proper metadata and with the proper name.
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    raw_data = pd.DataFrame({'wavelength': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency':
            exp_data['frequency']}
    exp.saveRawResults(raw_data, cond)
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
                       'temperature' + ns + '25.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        first_line = fh.readline()
        metadata_actual = literal_eval(first_line)

    metadata_desired = {'frequency': exp_data['frequency']}
    assert_equal(metadata_actual, metadata_desired)

def test_save_raw_results_data(exp, exp_data):
    """
    Tests that we correctly save the raw data and can read it out again.
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    data_desired = pd.DataFrame({'wavelength': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency':
            exp_data['frequency']}
    exp.saveRawResults(data_desired, cond)
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + 'temperature' + ns + '25.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        first_line = fh.readline()
        data_actual = pd.read_csv(fh)
    assert_allclose(data_actual, data_desired)
    assert_equal(data_actual.columns.values, data_desired.columns.values)

def test_save_raw_scalar(exp, exp_data):
    data_to_write = 4.05
    data_desired = pd.DataFrame({
            'wavelength': [1],
            'temperature': [25],
            'Value': [data_to_write]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency':
            exp_data['frequency']}
    filename_desired = 'TEST1.csv'
    exp.saveRawResults(data_to_write, cond)
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata_actual = literal_eval(fh.readline())
        data_actual = pd.read_csv(fh)
    assert_frame_equal(data_actual, data_desired)

def test_save_raw_scalar_multiple(exp, exp_data):
    data_to_write = 4.05
    data_desired = pd.DataFrame({
            'wavelength': [1, 2],
            'temperature': [25, 25],
            'Value': [data_to_write, data_to_write]})
    cond1 = {'wavelength': 1, 'temperature': 25, 'frequency':
            exp_data['frequency']}
    cond2 = {'wavelength': 2, 'temperature': 25, 'frequency':
            exp_data['frequency']}
    filename_desired = 'TEST1.csv'
    exp.saveRawResults(data_to_write, cond1)
    exp.saveRawResults(data_to_write, cond2)
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata_actual = literal_eval(fh.readline())
        data_actual = pd.read_csv(fh)
    assert_frame_equal(data_actual, data_desired)

def test_save_raw_scalar_multiple_units(ureg, exp_units, exp_data):
    data_to_write = 4.05*ureg.nA
    exp_units.measure_name = 'photocurrent'
    data_desired = pd.DataFrame({
            'wavelength (nm)': np.array([1, 2]),
            'temperature (K)': np.array([25, 25]),
            'photocurrent (nA)': [4.05, 4.05]})
    cond1 = {
        'wavelength': 1*ureg.nm,
        'temperature': 25*ureg.degK,
        'frequency': exp_data['frequency']*ureg.Hz}
    cond2 = {
        'wavelength': 2*ureg.nm,
        'temperature': 25*ureg.degK,
        'frequency': exp_data['frequency']*ureg.Hz}
    filename_desired = 'TEST1.csv'
    exp_units.saveRawResults(data_to_write, cond1)
    exp_units.saveRawResults(data_to_write, cond2)
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata_actual = literal_eval(fh.readline())
        data_actual = pd.read_csv(fh)
    assert_frame_equal(data_actual, data_desired)


@pytest.mark.skip
def testSaveDerivedQuantitiesFilename(exp, exp_data):
    master_data = {'TEST1': pd.DataFrame({'wavelength': [1, 2, 3],
                             'Mean': [1,2,4]})}
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    exp.master_data = master_data
    exp.saveDerivedQuantities()
    filename_desired = 'TEST1.csv'
    file_found = os.path.isfile(exp_data['data_full_path'] + filename_desired)
    assert_equal(file_found, True)

@pytest.mark.skip
def testSaveDerivedQuantitiesData(exp, exp_data):
    data_desired = pd.DataFrame({'wavelength': [1, 2, 3],
                             'Mean': [1,2,4]})
    exp.master_data = data_desired
    exp.saveDerivedQuantities()
    filename_desired = 'TEST1.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata = fh.readline()
        data_actual = pd.read_csv(fh)
    assert_allclose(data_actual, data_desired)
    assert_equal(data_actual.columns.values, data_desired.columns.values)

@pytest.mark.skip
def testSaveDerivedQuantitiesMetadata(exp, exp_data):
    data_desired = pd.DataFrame({'wavelength': [1, 2, 3],
                             'Mean': [1,2,4]})
    metadata_desired = {'frequency': exp_data['frequency']}
    exp.master_data = data_desired
    exp.saveDerivedQuantities()
    filename_desired = 'TEST1.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata_actual = literal_eval(fh.readline())
    assert_equal(metadata_actual, metadata_desired)
