"""
Tests data reading and writing operation, along with condition generation
"""
import pytest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment
from ast import literal_eval
from sciparse import assertDataDictEqual

@pytest.fixture
def exp_data(path_data):
        wavelength = np.array([1, 2])
        temperature = np.array([25, 50])
        replicate = np.array([0, 1])
        frequency = 8500
        fake_data = pd.DataFrame(
            {'Time (ms)': [0, 0.1, 0.2],
             'Voltage (V)': [1, 2, 3]})
        def data_func(cond):
            fake_data = pd.DataFrame(
                {'Time (ms)': [0, 0.1, 0.2],
                 'Voltage (V)': [1, 2, 3]})
            return fake_data
        yield dict({
            'wavelength': wavelength,
            'temperature': temperature,
            'replicate': replicate,
            'frequency': frequency,
            'fake_data': fake_data,
            'data_func': data_func,
            'major_separator': '~',
            'minor_separator': '=',
        }, **path_data)

@pytest.fixture
def exp(exp_data):
    exp = Experiment(
        name='TEST1', kind='test',
         measure_func=exp_data['data_func'],
         frequency=exp_data['frequency'],
         wavelength=exp_data['wavelength'],
         temperature=exp_data['temperature'],
         replicate=exp_data['replicate'])
    yield exp
    rmtree(exp_data['data_base_path'], ignore_errors=True)
    rmtree(exp_data['figures_base_path'], ignore_errors=True)
    rmtree(exp_data['designs_base_path'], ignore_errors=True)

def testExecuteExperimentFilesWritten(exp, exp_data, convert_name):
    """
    Tests that we successfully load a dataset
    """
    files = [
        convert_name('TEST1~wavelength-1~temperature-25~replicate-0'),
        convert_name('TEST1~wavelength-1~temperature-25~replicate-1'),
        convert_name('TEST1~wavelength-1~temperature-50~replicate-0'),
        convert_name('TEST1~wavelength-1~temperature-50~replicate-1'),
        convert_name('TEST1~wavelength-2~temperature-25~replicate-0'),
        convert_name('TEST1~wavelength-2~temperature-25~replicate-1'),
        convert_name('TEST1~wavelength-2~temperature-50~replicate-0'),
        convert_name('TEST1~wavelength-2~temperature-50~replicate-1')]
    full_filenames = [exp_data['data_full_path'] + fn + '.csv' for fn in files]
    exp.Execute()
    files_found = [os.path.isfile(fn) for fn in full_filenames]
    assert_equal(all(files_found), True)

def testExecuteExperimentDataCorrect(exp, exp_data, convert_name):
    exp.Execute()
    desired_data = {
        convert_name('TEST1~wavelength-1~temperature-25~replicate-0'): exp_data['fake_data'],
        convert_name('TEST1~wavelength-1~temperature-25~replicate-1'): exp_data['fake_data'],
        convert_name('TEST1~wavelength-1~temperature-50~replicate-0'):exp_data['fake_data'],
        convert_name('TEST1~wavelength-1~temperature-50~replicate-1'):exp_data['fake_data'],
        convert_name('TEST1~wavelength-2~temperature-25~replicate-0'):exp_data['fake_data'],
        convert_name('TEST1~wavelength-2~temperature-25~replicate-1'):exp_data['fake_data'],
        convert_name('TEST1~wavelength-2~temperature-50~replicate-0'):exp_data['fake_data'],
        convert_name('TEST1~wavelength-2~temperature-50~replicate-1'):exp_data['fake_data'],
    }
    actual_data = exp.data
    assertDataDictEqual(actual_data, desired_data)
