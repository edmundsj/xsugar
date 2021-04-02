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
from sciparse import assertDataDictEqual
from ast import literal_eval
from itertools import zip_longest
from pathlib import Path

def testLoadData(exp, exp_data, convert_name):
    data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                             'Mean': [1,2,4]})
    metadata_desired = {'non': 'sense'}
    filename = convert_name('TEST1~wavelengths-1~temperatures-25')
    file_extension = '.csv'
    full_filename = exp_data['data_full_path'] + filename + file_extension
    with open(full_filename, 'w+') as fh:
        fh.write(str(metadata_desired) + '\n')
        data_desired.to_csv(fh, mode='a', index=False)
    exp.loadData()
    data_actual = exp.data[filename]
    assert_frame_equal(data_actual, data_desired)

def testLoadXRDData(convert_name):
    file_location = os.path.dirname(os.path.abspath(__file__))
    base_path = file_location + '/data'
    exp = Experiment(name='TEST1', kind='test', base_path=base_path)
    exp.loadXRDData()
    data_desired = pd.DataFrame({
        'Angle (deg)': [69.05, 69.055, 69.06, 69.065, 69.07,69.075,69.08,
        69.085, 69.09, 69.095, 69.1, 69.105, 69.11, 69.115],
        'Counts': [24, 30, 28, 40, 132, 272, 3472, 16368,21970,10562,
                   1210,264,130,64]})
    data_actual = exp.data[convert_name('TEST1~1~type-locked_coupled~peak-Si')]
    assert_frame_equal(data_actual, data_desired)

def testLoadXRDMetadata(convert_name):
    file_location = os.path.dirname(os.path.abspath(__file__))
    base_path = file_location + '/data'
    exp = Experiment(name='TEST1', kind='test', base_path=base_path)
    exp.loadXRDData()
    metadata_desired = {
        'date': '02/10/2021',
        'increment': 0.005, 'scantype': 'locked coupled',
        'start': 69.05, 'steps': 14, 'time': 1,
        'theta': 34.0, '2theta': 68.0, 'phi': 180.13, 'chi': -0.972}
    metadata_actual = exp.metadata[convert_name('TEST1~1~type-locked_coupled~peak-Si')]
    assert_equal(metadata_actual, metadata_desired)

def testLoadConstants(exp, exp_data, convert_name):
    """
    Tests that we can load metadata from a file successfully
    """
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    with open(exp_data['data_full_path'] + \
              convert_name('TEST1~wavelengths-1~temperatures-25.csv'), 'w+') as fh:
        fh.write('{"frequency": 8500}\n')
        fh.write(f'Time, Data\n')
        fh.write(f'1, 2\n')
    exp = Experiment(name='TEST1', kind='test')
    exp.loadData()
    constants_actual = exp.constants
    constants_desired = {'frequency': frequency}
    assert_equal(constants_actual, constants_desired)

def testLoadMetadata(exp, exp_data, convert_name):
    """
    Tests that we can load metadata from a file successfully
    """
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    with open(exp_data['data_full_path'] + \
              convert_name('TEST1~wavelengths-1~temperatures-25.csv'), 'w+') as fh:
        fh.write('{"frequency": 8500}\n')
        fh.write(f'Time, Data\n')
        fh.write(f'1, 2\n')
    exp = Experiment(name='TEST1', kind='test')
    exp.loadData()
    metadata_actual = exp.metadata
    metadata_desired = {
        convert_name('TEST1~wavelengths-1~temperatures-25'): \
                        {'frequency': frequency}}
    assert_equal(metadata_actual, metadata_desired)

# TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
# THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

# TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
# THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

def testLookup(exp, convert_name):
    fudge_data = pd.DataFrame(
        {'Time (ms)': [1, 2, 3],
        'Voltage (V)': [0,0.1, 0.2]})
    exp.data = {
        convert_name('TEST1~wavelengths-1~temperatures-25'):fudge_data,
        convert_name('TEST1~wavelengths-2~temperatures-25'):fudge_data,
        convert_name('TEST1~wavelengths-2~temperatures-35'):fudge_data,
        convert_name('TEST1~wavelengths-2~temperatures-35'):fudge_data,
                    }
    data_actual = exp.lookup(temperatures=25)
    data_desired = {
        convert_name('TEST1~wavelengths-1~temperatures-25'):fudge_data,
        convert_name('TEST1~wavelengths-2~temperatures-25'):fudge_data,}
    assertDataDictEqual(data_actual, data_desired)
