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
from xsugar import assertDataDictEqual

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


def testGenerateMasterData1Var(exp):
    scalar_data = {'TEST1~wavelength-1~temperature-25': 1,
                   'TEST1~wavelength-1~temperature-35': 2}
    desired_data = pd.DataFrame({
        'temperature': [25, 35],
        'Value': [1, 2]})
    actual_data = exp.master_data(data_dict=scalar_data)
    assert_frame_equal(actual_data, desired_data)

def testGenerateMasterData2Var(exp):
    scalar_data = {'TEST1~wavelength-1~temperature-25': 1,
                   'TEST1~wavelength-1~temperature-35': 2,
                   'TEST1~wavelength-2~temperature-25': 3,
                   'TEST1~wavelength-2~temperature-35': 4}
    desired_data = pd.DataFrame({
        'wavelength': [1, 1, 2, 2],
        'temperature': [25, 35, 25, 35],
        'Value': [1, 2, 3, 4]})
    actual_data = exp.master_data(data_dict=scalar_data)
    assert_frame_equal(actual_data, desired_data)

def test_data_from_master(exp):
    master_data = pd.DataFrame({
        'wavelength': [1, 2],
        'Value': [3, 4]})
    desired_data = {
        'TEST1~wavelength-1': 3,
        'TEST1~wavelength-2': 4,
    }
    actual_data = exp.data_from_master(master_data)
    assertDataDictEqual(actual_data, desired_data)

def test_data_from_master_2var(exp):
    master_data = pd.DataFrame({
        'wavelength': [0, 1, 2, 0, 1, 2],
        'temperature': [25.0, 25.0, 25.0, 35.0, 35.0, 35.0],
        'Value': [0, 1, 2, 3, 4, 5]})
    desired_data_dict = {
        'TEST1~wavelength-0~temperature-25.0': 0,
        'TEST1~wavelength-1~temperature-25.0': 1,
        'TEST1~wavelength-2~temperature-25.0': 2,
        'TEST1~wavelength-0~temperature-35.0': 3,
        'TEST1~wavelength-1~temperature-35.0': 4,
        'TEST1~wavelength-2~temperature-35.0': 5}
    actual_data_dict = exp.data_from_master(master_data)
    assertDataDictEqual(actual_data_dict, desired_data_dict)

def testGenerateMasterDataDict1Var(exp):
    data_dict = {'TEST1~wavelength-1': 3.0,
        'TEST1~wavelength-2': 4.0}
    desired_data = {
        'TEST1~wavelength-all': pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [3.0, 4.0]})}
    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)

def testGenerateMasterDataDict2Var(exp):
    master_data = pd.DataFrame({
        'wavelength': [0, 1, 2, 0, 1, 2],
        'temperature': [25.0, 25.0, 25.0, 35.0, 35.0, 35.0],
        'Value': [0, 1, 2, 3, 4, 5]})
    data_dict = {
        'TEST1~wavelength-0~temperature-25.0': 0,
        'TEST1~wavelength-1~temperature-25.0': 1,
        'TEST1~wavelength-2~temperature-25.0': 2,
        'TEST1~wavelength-0~temperature-35.0': 3,
        'TEST1~wavelength-1~temperature-35.0': 4,
        'TEST1~wavelength-2~temperature-35.0': 5}
    desired_data = {
        'TEST1~wavelength-all':
        {
            'TEST1~temperature-25.0': pd.DataFrame({
                'wavelength': [0, 1, 2],
                'Value': [0, 1, 2]}),
            'TEST1~temperature-35.0': pd.DataFrame({
                'wavelength': [0,1,2],
                'Value': [3, 4, 5]})
        },
        'TEST1~temperature-all':
        {
            'TEST1~wavelength-0': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [0, 3]}),
            'TEST1~wavelength-1': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [1, 4] }),
            'TEST1~wavelength-2': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [2, 5] })
        },
    }

    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)

def testGenerateMasterDataDict3Var(exp):
    master_data = pd.DataFrame({
        'wavelength': [0, 0, 0, 0, 1, 1, 1, 1],
        'temperature': [25.0, 25.0, 35.0, 35.0, 25.0, 25.0, 35.0, 35.0],
        'material': ['Au', 'Al', 'Au', 'Al', 'Au', 'Al', 'Au', 'Al'],
        'Value': [0, 1, 2, 3, 4, 5, 6, 7]})
    data_dict = {
        'TEST1~wavelength-0~temperature-25.0~material-Au': 0,
        'TEST1~wavelength-0~temperature-25.0~material-Al': 1,
        'TEST1~wavelength-0~temperature-35.0~material-Au': 2,
        'TEST1~wavelength-0~temperature-35.0~material-Al': 3,
        'TEST1~wavelength-1~temperature-25.0~material-Au': 4,
        'TEST1~wavelength-1~temperature-25.0~material-Al': 5,
        'TEST1~wavelength-1~temperature-35.0~material-Au': 6,
        'TEST1~wavelength-1~temperature-35.0~material-Al': 7}
    desired_data = {
        'TEST1~wavelength-all~material-Au':
        {
            'TEST1~temperature-25.0': pd.DataFrame({
                'wavelength': [0, 1],
                'Value': [0, 4]}),
            'TEST1~temperature-35.0': pd.DataFrame({
                'wavelength': [0, 1],
                'Value': [2, 6]}),
        },
        'TEST1~wavelength-all~material-Al':
        {
            'TEST1~temperature-25.0': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [1,5]}),
            'TEST1~temperature-35.0': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [3,7]}),
        },
        'TEST1~wavelength-all~temperature-25.0': {
            'TEST1~material-Au': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [0,4]}),
            'TEST1~material-Al': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [1,5]}),
        },
        'TEST1~wavelength-all~temperature-35.0': {
            'TEST1~material-Au': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [2,6]}),
            'TEST1~material-Al': pd.DataFrame({
                'wavelength': [0,1],
                'Value': [3,7]}),
        },
        'TEST1~temperature-all~material-Au': {
            'TEST1~wavelength-0': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [0,2]}),
            'TEST1~wavelength-1': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [4,6]}),
        },
        'TEST1~temperature-all~material-Al': {
            'TEST1~wavelength-0': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [1,3]}),
            'TEST1~wavelength-1': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [5,7]}),
        },
        'TEST1~temperature-all~wavelength-0': {
            'TEST1~material-Au': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [0,2]}),
            'TEST1~material-Al': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [1,3]}),
        },
        'TEST1~temperature-all~wavelength-1': {
            'TEST1~material-Au': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [4,6]}),
            'TEST1~material-Al': pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [5,7]}),
        },
        'TEST1~material-all~temperature-25.0': {
            'TEST1~wavelength-0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [0,1]}),
            'TEST1~wavelength-1': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [4,5]}),
        },
        'TEST1~material-all~temperature-35.0': {
            'TEST1~wavelength-0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [2,3]}),
            'TEST1~wavelength-1': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [6,7]}),
        },
        'TEST1~material-all~wavelength-0': {
            'TEST1~temperature-25.0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [0,1]}),
            'TEST1~temperature-35.0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [2,3]}),
        },
        'TEST1~material-all~wavelength-1': {
            'TEST1~temperature-25.0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [4,5]}),
            'TEST1~temperature-35.0': pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [6,7]}),
        },
    }

    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)
