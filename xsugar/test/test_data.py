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
from xsugar import Experiment, assertDataDictEqual
from ast import literal_eval
from itertools import zip_longest


def testExtractConstants(exp, exp_data):
    """
    Tests that our experiment properly extracts metadata from a list of
    arbitrary keyword arguments.
    """
    actual_constants = exp.constants
    desired_constants = {'frequency': exp_data['frequency']}
    assert_equal(actual_constants , desired_constants)


def test_extract_factors(exp, exp_data):
    """
    Tests proper extraction of conditions from arbitrary input keyword
    arguments.
    """
    actual_factors = exp.factors
    desired_factors = {
        'wavelength': exp_data['wavelength'],
        'temperature': exp_data['temperature']}
    assert_equal(actual_factors, desired_factors)

def test_generate_conditions(exp, exp_data):
    """
    Tests that our generator actually generates all the right combinations
    and in the right order.
    """
    expected_conds = [
        {'wavelength': 1, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 1, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 50, 'frequency': 8500},
    ]
    for actual_cond, desired_cond in  zip(exp.conditions,
                                          expected_conds):
        assert_equal(actual_cond, desired_cond)

def testSaveRawResultsFilename(exp, exp_data):
    """
    Tests that we save the raw results in the proper directory and with
    the proper metadata and with the proper name.
    """
    raw_data = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelengths': 1, 'temperatures': 25, 'frequency': 8500}
    exp.saveRawResults(raw_data, cond)
    filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
    file_found = os.path.isfile(exp_data['data_full_path'] + filename_desired)
    assert_equal(file_found, True)

def testSaveRawResultsMetadata(exp, exp_data):
    """
    Tests that we save the raw results in the proper directory and with
    the proper metadata and with the proper name.
    """
    raw_data = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelengths': 1, 'temperatures': 25, 'frequency':
            exp_data['frequency']}
    exp.saveRawResults(raw_data, cond)
    filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        first_line = fh.readline()
        metadata_actual = literal_eval(first_line)

    metadata_desired = {'frequency': exp_data['frequency']}
    assert_equal(metadata_actual, metadata_desired)

def testSaveRawResultsData(exp, exp_data):
    """
    Tests that we correctly save the raw data and can read it out again.
    """
    data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Current': [4,4.5,6]})
    cond = {'wavelengths': 1, 'temperatures': 25, 'frequency':
            exp_data['frequency']}
    exp.saveRawResults(data_desired, cond)
    filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        first_line = fh.readline()
        data_actual = pd.read_csv(fh)
    assert_allclose(data_actual, data_desired)
    assert_equal(data_actual.columns.values, data_desired.columns.values)

@pytest.mark.skip
def testSaveDerivedQuantitiesFilename(exp, exp_data):
    master_data = {'TEST1': pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Mean': [1,2,4]})}
    cond = {'wavelengths': 1, 'temperatures': 25, 'frequency': 8500}
    exp.master_data = master_data
    exp.saveDerivedQuantities()
    filename_desired = 'TEST1.csv'
    file_found = os.path.isfile(exp_data['data_full_path'] + filename_desired)
    assert_equal(file_found, True)

@pytest.mark.skip
def testSaveDerivedQuantitiesData(exp, exp_data):
    data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
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
    data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Mean': [1,2,4]})
    metadata_desired = {'frequency': exp_data['frequency']}
    exp.master_data = data_desired
    exp.saveDerivedQuantities()
    filename_desired = 'TEST1.csv'
    with open(exp_data['data_full_path'] + filename_desired) as fh:
        metadata_actual = literal_eval(fh.readline())
    assert_equal(metadata_actual, metadata_desired)

def testLoadData(exp, exp_data):
    data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Mean': [1,2,4]})
    metadata_desired = {'non': 'sense'}
    filename = 'TEST1~wavelengths-1~temperatures-25'
    file_extension = '.csv'
    full_filename = exp_data['data_full_path'] + filename + file_extension
    with open(full_filename, 'w+') as fh:
        fh.write(str(metadata_desired) + '\n')
        data_desired.to_csv(fh, mode='a', index=False)
    exp.loadData()
    data_actual = exp.data[filename]
    assert_frame_equal(data_actual, data_desired)

def testLoadXRDData():
    exp = Experiment(name='TEST1', kind='test_permanent')
    exp.loadXRDData()
    data_desired = pd.DataFrame({
        'Angle (deg)': [69.05, 69.055, 69.06, 69.065, 69.07,69.075,69.08,
        69.085, 69.09, 69.095, 69.1, 69.105, 69.11, 69.115],
        'Counts': [24, 30, 28, 40, 132, 272, 3472, 16368,21970,10562,
                   1210,264,130,64]})
    data_actual = exp.data['TEST1~1~type-locked_coupled~peak-Si']
    assert_frame_equal(data_actual, data_desired)

def testLoadXRDMetadata():
    exp = Experiment(name='TEST1', kind='test_permanent')
    exp.loadXRDData()
    metadata_desired = {
        'date': '02/10/2021',
        'increment': 0.005, 'scantype': 'locked coupled',
        'start': 69.05, 'steps': 14, 'time': 1,
        'theta': 34.0, '2theta': 68.0, 'phi': 180.13, 'chi': -0.972}
    metadata_actual = exp.metadata['TEST1~1~type-locked_coupled~peak-Si']
    assert_equal(metadata_actual, metadata_desired)

def testLoadConstants(exp, exp_data):
    """
    Tests that we can load metadata from a file successfully
    """
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    with open(exp_data['data_full_path'] + \
              'TEST1~wavelengths-1~temperatures-25.csv', 'w+') as fh:
        fh.write('{"frequency": 8500}\n')
        fh.write(f'Time, Data\n')
        fh.write(f'1, 2\n')
    exp = Experiment(name='TEST1', kind='test')
    exp.loadData()
    constants_actual = exp.constants
    constants_desired = {'frequency': frequency}
    assert_equal(constants_actual, constants_desired)

def testLoadMetadata(exp, exp_data):
    """
    Tests that we can load metadata from a file successfully
    """
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    with open(exp_data['data_full_path'] + \
              'TEST1~wavelengths-1~temperatures-25.csv', 'w+') as fh:
        fh.write('{"frequency": 8500}\n')
        fh.write(f'Time, Data\n')
        fh.write(f'1, 2\n')
    exp = Experiment(name='TEST1', kind='test')
    exp.loadData()
    metadata_actual = exp.metadata
    metadata_desired = {'TEST1~wavelengths-1~temperatures-25': \
                        {'frequency': frequency}}
    assert_equal(metadata_actual, metadata_desired)

# TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
# THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

# TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
# THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

def testLookup(exp):
    fudge_data = pd.DataFrame(
        {'Time (ms)': [1, 2, 3],
        'Voltage (V)': [0,0.1, 0.2]})
    exp.data = {'TEST1~wavelengths-1~temperatures-25':fudge_data,
                     'TEST1~wavelengths-2~temperatures-25':fudge_data,
                     'TEST1~wavelengths-2~temperatures-35':fudge_data,
                     'TEST1~wavelengths-2~temperatures-35':fudge_data,
                    }
    data_actual = exp.lookup(temperatures=25)
    data_desired = {'TEST1~wavelengths-1~temperatures-25':fudge_data,
                     'TEST1~wavelengths-2~temperatures-25':fudge_data,}
    assertDataDictEqual(data_actual, data_desired)
