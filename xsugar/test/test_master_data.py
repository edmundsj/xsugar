import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment, ureg
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


def testGenerateMasterData1Var(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    name1 = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
        'temperature' + ns + '25'
    name2 = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
        'temperature' + ns + '35'
    scalar_data = {
        name1: 1,
        name2: 2,
    }
    desired_data = pd.DataFrame({
        'temperature': [25, 35],
        'Value': [1, 2]})
    actual_data = exp.master_data(data_dict=scalar_data)
    assert_frame_equal(actual_data, desired_data)

def test_master_data_units(exp_units, convert_name):
    name = convert_name('TEST1~wavelength=25nm~temperature=305K')
    data_dict = {name: ureg.mV * 1.5}
    desired_master_data = pd.DataFrame({
            'wavelength (nm)': [25],
            'temperature (K)': [305],
            'voltage (mV)': [1.5]})
    actual_master_data = exp_units.master_data(data_dict)
    assert_frame_equal(actual_master_data, desired_master_data)

def test_master_data_nodrop(exp_units, convert_name):
    """
    Checks that we do not drop the last column.
    """
    name1 = convert_name('TEST1~wavelength=25nm~temperature=305K')
    name2 = convert_name('TEST1~wavelength=35nm~temperature=306K')
    data_dict = {
        name1: ureg.mV * 1.5,
        name2: ureg.mV * 1.5,
    }
    desired_master_data = pd.DataFrame({
            'wavelength (nm)': [25, 35],
            'temperature (K)': [305, 306],
            'voltage (mV)': [1.5, 1.5]})
    actual_master_data = exp_units.master_data(data_dict)
    assert_frame_equal(actual_master_data, desired_master_data)

def testGenerateMasterData2Var(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    name1 = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
        'temperature' + ns + '25'
    name2 = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
        'temperature' + ns + '35'
    name3 = 'TEST1' + js + 'wavelength' + ns + '2' + js + \
        'temperature' + ns + '25'
    name4 = 'TEST1' + js + 'wavelength' + ns + '2' + js + \
        'temperature' + ns + '35'
    scalar_data = {
               name1: 1,
               name2: 2,
               name3: 3,
               name4: 4}

    desired_data = pd.DataFrame({
        'wavelength': [1, 1, 2, 2],
        'temperature': [25, 35, 25, 35],
        'Value': [1, 2, 3, 4]})
    actual_data = exp.master_data(data_dict=scalar_data)
    assert_frame_equal(actual_data, desired_data)

def test_data_from_master(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    master_data = pd.DataFrame({
        'wavelength': [1, 2],
        'Value': [3, 4]})
    name1 = 'TEST1' + js + 'wavelength' + ns + '1'
    name2 = 'TEST1' + js + 'wavelength' + ns + '2'
    desired_data = {
        name1: 3,
        name2: 4
    }
    actual_data = exp.data_from_master(master_data)
    assertDataDictEqual(actual_data, desired_data)

def test_data_from_master_units(exp_units, convert_name):
    desired_name = convert_name('TEST1~wavelength=25nm~temperature=305K')
    master_data = pd.DataFrame({
            'wavelength (nm)': [25],
            'temperature (K)': [305],
            'voltage (mV)': [1.5]})
    desired_data = {desired_name: 1.5 * ureg.mV}
    actual_data = exp_units.data_from_master(master_data)
    assertDataDictEqual(actual_data, desired_data)

def test_data_from_master_2var(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    master_data = pd.DataFrame({
        'wavelength': [0, 1, 2, 0, 1, 2],
        'temperature': [25.0, 25.0, 25.0, 35.0, 35.0, 35.0],
        'Value': [0, 1, 2, 3, 4, 5]})
    names = [
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '2' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '35.0',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '35.0',
        'TEST1' + js + 'wavelength' + ns + '2' + js + \
            'temperature' + ns + '35.0',
    ]
    desired_data_dict = {
        names[0]: 0,
        names[1]: 1,
        names[2]: 2,
        names[3]: 3,
        names[4]: 4,
        names[5]: 5}
    actual_data_dict = exp.data_from_master(master_data)
    assertDataDictEqual(actual_data_dict, desired_data_dict)

def testGenerateMasterDataDict1Var(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    name1 = 'TEST1' + js + 'wavelength' + ns + '1'
    name2 = 'TEST1' + js + 'wavelength' + ns + '2'
    name_all = 'TEST1' + js + 'wavelength' + ns + 'all'
    data_dict = {
        name1: 3.0,
        name2: 4.0}
    desired_data = {
        name_all: pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [3.0, 4.0]})}
    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_1var_units(exp_units, convert_name):
    name1 = convert_name('TEST1~wavelength=1nm')
    name2 = convert_name('TEST1~wavelength=2nm')
    name_all = convert_name('TEST1~wavelength=all')
    data_dict = {
        name1: 3.0 * ureg.nA,
        name2: 4.0 * ureg.nA}
    desired_data = {
        name_all: pd.DataFrame({
                'wavelength (nm)': [1, 2],
                'current (nA)': [3.0, 4.0]})}
    actual_data = exp_units.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)

def testGenerateMasterDataDict2Var(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    master_data = pd.DataFrame({
        'wavelength': [0, 1, 2, 0, 1, 2],
        'temperature': [25.0, 25.0, 25.0, 35.0, 35.0, 35.0],
        'Value': [0, 1, 2, 3, 4, 5]})
    names = [
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '2' + js + \
            'temperature' + ns + '25.0',
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '35.0',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '35.0',
        'TEST1' + js + 'wavelength' + ns + '2' + js + \
            'temperature' + ns + '35.0',
    ]
    data_dict = {
        names[0]: 0,
        names[1]: 1,
        names[2]: 2,
        names[3]: 3,
        names[4]: 4,
        names[5]: 5}
    desired_data = {
        'TEST1' + js + 'wavelength' + ns + 'all':
        {
            'TEST1' + js + 'temperature' + ns + '25.0': pd.DataFrame({
                'wavelength': [0, 1, 2],
                'Value': [0, 1, 2]}),
            'TEST1' + js + 'temperature' + ns + '35.0': pd.DataFrame({
                'wavelength': [0,1,2],
                'Value': [3, 4, 5]})
        },
        'TEST1' + js + 'temperature' + ns + 'all':
        {
            'TEST1' + js + 'wavelength' + ns + '0': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [0, 3]}),
            'TEST1' + js + 'wavelength' + ns + '1': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [1, 4] }),
            'TEST1' + js + 'wavelength' + ns + '2': pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [2, 5] })
        },
    }

    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_includue_x(exp, exp_data, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1~temperature=25.0',
        'TEST1~wavelength=2~temperature=25.0',
        'TEST1~wavelength=1~temperature=35.0',
        'TEST1~wavelength=2~temperature=35.0',
        ]]
    name_all = convert_name('TEST1~wavelength=all')
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
        names[2]: 3.0,
        names[3]: 4.0,
    }
    desired_data = {
        convert_name('TEST1~wavelength=all'): {
            convert_name('TEST1~temperature=25.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [1.0, 2.0]}),

            convert_name('TEST1~temperature=35.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [3.0, 4.0]}),
        },
    }
    actual_data = exp.master_data_dict(
            data_dict, x_axis_include=['wavelength'])
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_exclude_x(exp, exp_data, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1~temperature=25.0',
        'TEST1~wavelength=2~temperature=25.0',
        'TEST1~wavelength=1~temperature=35.0',
        'TEST1~wavelength=2~temperature=35.0',
        ]]
    name_all = convert_name('TEST1~temperature=all')
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
        names[2]: 3.0,
        names[3]: 4.0,
    }
    desired_data = {
        convert_name('TEST1~temperature=all'): {
            convert_name('TEST1~wavelength=1'):
                pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [1.0, 3.0]}),

            convert_name('TEST1~wavelength=2'):
                pd.DataFrame({
                'temperature': [25.0, 35.0],
                'Value': [2.0, 4.0]}),
        },
    }
    actual_data = exp.master_data_dict(
            data_dict, x_axis_exclude=['wavelength'])
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_includue_c(exp, exp_data, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1~temperature=25.0',
        'TEST1~wavelength=2~temperature=25.0',
        'TEST1~wavelength=1~temperature=35.0',
        'TEST1~wavelength=2~temperature=35.0',
        ]]
    name_all = convert_name('TEST1~wavelength=all')
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
        names[2]: 3.0,
        names[3]: 4.0,
    }
    desired_data = {
        convert_name('TEST1~wavelength=all'): {
            convert_name('TEST1~temperature=25.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [1.0, 2.0]}),

            convert_name('TEST1~temperature=35.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [3.0, 4.0]}),
        },
    }
    actual_data = exp.master_data_dict(
            data_dict, c_axis_include=['temperature'])
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_exclude_c(exp, exp_data, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1~temperature=25.0',
        'TEST1~wavelength=2~temperature=25.0',
        'TEST1~wavelength=1~temperature=35.0',
        'TEST1~wavelength=2~temperature=35.0',
        ]]
    name_all = convert_name('TEST1~wavelength=all')
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
        names[2]: 3.0,
        names[3]: 4.0,
    }
    desired_data = {
        convert_name('TEST1~wavelength=all'): {
            convert_name('TEST1~temperature=25.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [1.0, 2.0]}),

            convert_name('TEST1~temperature=35.0'):
                pd.DataFrame({
                'wavelength': [1, 2],
                'Value': [3.0, 4.0]}),
        },
    }
    actual_data = exp.master_data_dict(
            data_dict, c_axis_exclude=['wavelength'])
    assertDataDictEqual(actual_data, desired_data)

def testGenerateMasterDataDict3Var(exp, exp_data, convert_name):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    master_data = pd.DataFrame({
        'wavelength': [0, 0, 0, 0, 1, 1, 1, 1],
        'temperature': [25.0, 25.0, 35.0, 35.0, 25.0, 25.0, 35.0, 35.0],
        'material': ['Au', 'Al', 'Au', 'Al', 'Au', 'Al', 'Au', 'Al'],
        'Value': [0, 1, 2, 3, 4, 5, 6, 7]})
    names = [
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '25.0' + js + 'material' + ns + 'Au',
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '25.0' + js + 'material' + ns + 'Al',
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '35.0' + js + 'material' + ns + 'Au',
        'TEST1' + js + 'wavelength' + ns + '0' + js + \
            'temperature' + ns + '35.0' + js + 'material' + ns + 'Al',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '25.0' + js + 'material' + ns + 'Au',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '25.0' + js + 'material' + ns + 'Al',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '35.0' + js + 'material' + ns + 'Au',
        'TEST1' + js + 'wavelength' + ns + '1' + js + \
            'temperature' + ns + '35.0' + js + 'material' + ns + 'Al',
    ]
    data_dict = {
        names[0]: 0,
        names[1]: 1,
        names[2]: 2,
        names[3]: 3,
        names[4]: 4,
        names[5]: 5,
        names[6]: 6,
        names[7]: 7}
    desired_data = {
        'TEST1' + js + 'wavelength' + ns + 'all' + js + 'material' + \
            ns + 'Au': {
            convert_name('TEST1~temperature-25.0'): pd.DataFrame({
                'wavelength': [0, 1],
                'Value': [0, 4]}),
            convert_name('TEST1~temperature-35.0'): pd.DataFrame({
                'wavelength': [0, 1],
                'Value': [2, 6]}),
        },
        convert_name('TEST1~wavelength-all~material-Al'):
        {
            convert_name('TEST1~temperature-25.0'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [1,5]}),
            convert_name('TEST1~temperature-35.0'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [3,7]}),
        },
        convert_name('TEST1~wavelength-all~temperature-25.0'): {
            convert_name('TEST1~material-Au'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [0,4]}),
            convert_name('TEST1~material-Al'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [1,5]}),
        },
        convert_name('TEST1~wavelength-all~temperature-35.0'): {
            convert_name('TEST1~material-Au'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [2,6]}),
            convert_name('TEST1~material-Al'): pd.DataFrame({
                'wavelength': [0,1],
                'Value': [3,7]}),
        },
        convert_name('TEST1~temperature-all~material-Au'): {
            convert_name('TEST1~wavelength-0'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [0,2]}),
            convert_name('TEST1~wavelength-1'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [4,6]}),
        },
        convert_name('TEST1~temperature-all~material-Al'): {
            convert_name('TEST1~wavelength-0'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [1,3]}),
            convert_name('TEST1~wavelength-1'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [5,7]}),
        },
        convert_name('TEST1~temperature-all~wavelength-0'): {
            convert_name('TEST1~material-Au'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [0,2]}),
            convert_name('TEST1~material-Al'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [1,3]}),
        },
        convert_name('TEST1~temperature-all~wavelength-1'): {
            convert_name('TEST1~material-Au'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [4,6]}),
            convert_name('TEST1~material-Al'): pd.DataFrame({
                'temperature': [25.0,35.0],
                'Value': [5,7]}),
        },
        convert_name('TEST1~material-all~temperature-25.0'): {
            convert_name('TEST1~wavelength-0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [0,1]}),
            convert_name('TEST1~wavelength-1'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [4,5]}),
        },
        convert_name('TEST1~material-all~temperature-35.0'): {
            convert_name('TEST1~wavelength-0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [2,3]}),
            convert_name('TEST1~wavelength-1'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [6,7]}),
        },
        convert_name('TEST1~material-all~wavelength-0'): {
            convert_name('TEST1~temperature-25.0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [0,1]}),
            convert_name('TEST1~temperature-35.0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [2,3]}),
        },
        convert_name('TEST1~material-all~wavelength-1'): {
            convert_name('TEST1~temperature-25.0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [4,5]}),
            convert_name('TEST1~temperature-35.0'): pd.DataFrame({
                'material': ['Au', 'Al'],
                'Value': [6,7]}),
        },
    }

    actual_data = exp.master_data_dict(data_dict)
    assertDataDictEqual(actual_data, desired_data)
