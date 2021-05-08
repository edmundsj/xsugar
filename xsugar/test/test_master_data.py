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
    exp = Experiment(name='TEST1', kind='test')
    yield exp
    rmtree(path_data['data_base_path'], ignore_errors=True)
    rmtree(path_data['figures_base_path'], ignore_errors=True)
    rmtree(path_data['designs_base_path'], ignore_errors=True)

@pytest.fixture
def data_1x2():
    index_values = (1, 2)
    index = pd.Index(index_values, name='wavelength')
    data = pd.DataFrame(index=index,
            data={'current (nA)': [0, 1]})
    yield data

@pytest.fixture
def data_1x2_units():
    index_values = (1, 2)
    index = pd.Index(index_values, name='wavelength (nm)')
    data = pd.DataFrame(index=index,
            data={'current (nA)': [0, 1]})
    yield data

@pytest.fixture
def data_2x1():
    index_values = ((1, 25), (1, 35))
    index = pd.MultiIndex.from_tuples(index_values,
            names=('wavelength', 'temperature'))
    data = pd.DataFrame(index=index,
            data={'current (nA)': [0, 1]})
    yield data

@pytest.fixture
def data_2x2():
    index_values = ((1, 25.0), (1, 35.0), (2, 25.0), (2, 35.0))
    index = pd.MultiIndex.from_tuples(index_values,
            names=('wavelength', 'temperature'))
    data = pd.DataFrame(index=index,
            data={'current (nA)': [0, 1, 2, 3]})
    yield data

@pytest.fixture
def data_dict_2x2(data_2x2, convert_name):
    desired_data = {
        convert_name('TEST1~temperature=c~wavelength=x'): {
            convert_name('TEST1~temperature=25.0~wavelength=x'):
                pd.DataFrame(
                index=pd.Index([1,2], name='wavelength'),
                data={'current (nA)': [0, 2]}),

            convert_name('TEST1~temperature=35.0~wavelength=x'):
                pd.DataFrame(
                index=pd.Index([1,2], name='wavelength'),
                data={'current (nA)': [1, 3]}),
        },
    }
    yield desired_data

@pytest.fixture
def data_3x2():
    wavelengths = (0, 0, 0, 0, 1, 1, 1, 1)
    temperatures = (25.0, 25.0, 35.0, 35.0, 25.0, 25.0, 35.0, 35.0)
    materials = ('Au', 'Al', 'Au', 'Al', 'Au', 'Al', 'Au', 'Al')
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    index = pd.MultiIndex.from_arrays(
            (wavelengths, temperatures, materials),
            names = ('wavelength', 'temperature', 'material'))
    data = pd.DataFrame(index=index,
            data={'Value': values})
    yield data

def test_master_data_dict_1x2(data_1x2, exp, convert_name):
    actual_name = convert_name('TEST1~wavelength=x')
    desired_dict = {actual_name: data_1x2}
    actual_dict = exp.master_data_dict(data_1x2)
    assertDataDictEqual(actual_dict, desired_dict)

def test_master_data_dict_2x2(exp, data_2x2, convert_name):
    desired_dict = {
        convert_name('TEST1~temperature=c~wavelength=x'):
        {
            convert_name('TEST1~temperature=25.0~wavelength=x'):
                pd.DataFrame(
                index=pd.Index([1, 2], name='wavelength'),
                data={'current (nA)': [0, 2]}),
            convert_name('TEST1~temperature=35.0~wavelength=x'):
            pd.DataFrame(
                index=pd.Index([1, 2], name='wavelength'),
                data={'current (nA)': [1, 3]})
        },
        convert_name('TEST1~temperature=x~wavelength=c'):
        {
            convert_name('TEST1~temperature=x~wavelength=1'):
            pd.DataFrame(
                index=pd.Index([25.0, 35.0], name='temperature'),
                data={'current (nA)': [0, 1]}),
            convert_name('TEST1~temperature=x~wavelength=2'):
            pd.DataFrame(
                index=pd.Index([25.0, 35.0], name='temperature'),
                data={'current (nA)': [2, 3] }),
        },
    }
    actual_dict = exp.master_data_dict(data_2x2)

    assertDataDictEqual(actual_dict, desired_dict)

def test_master_data_dict_includue_x(exp, data_2x2, data_dict_2x2):
    actual_data = exp.master_data_dict(
            data=data_2x2, x_axis_include=['wavelength'])
    desired_data = data_dict_2x2
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_exclude_x(exp, data_2x2, data_dict_2x2):
    actual_data = exp.master_data_dict(
            data_2x2, x_axis_exclude=['temperature'])
    desired_data = data_dict_2x2
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_includue_c(exp, data_2x2, data_dict_2x2):
    actual_data = exp.master_data_dict(
            data_2x2, c_axis_include=['temperature'])
    desired_data = data_dict_2x2
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_exclude_c(exp, data_2x2, data_dict_2x2):
    actual_data = exp.master_data_dict(
            data_2x2, c_axis_exclude=['wavelength'])
    desired_data = data_dict_2x2
    assertDataDictEqual(actual_data, desired_data)

def test_master_data_dict_3var(exp, data_3x2):
    desired_data =  \
    {
        'TEST1~material=Au~temperature=c~wavelength=x':
            {'TEST1~material=Au~temperature=25.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [0, 4]}),
            'TEST1~material=Au~temperature=35.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [2, 6]}),
            },
        'TEST1~material=Al~temperature=c~wavelength=x':
            {'TEST1~material=Al~temperature=25.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [1, 5]}),
            'TEST1~material=Al~temperature=35.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [3, 7]}),
            },
        'TEST1~material=c~temperature=25.0~wavelength=x':
            {'TEST1~material=Au~temperature=25.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [0, 4]}),
            'TEST1~material=Al~temperature=25.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [1, 5]}),
            },
        'TEST1~material=c~temperature=35.0~wavelength=x':
            {'TEST1~material=Au~temperature=35.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [2, 6]}),
            'TEST1~material=Al~temperature=35.0~wavelength=x':
                pd.DataFrame(
                        index=pd.Index([0, 1], name='wavelength'),
                        data={'Value': [3, 7]}),
            },
        'TEST1~material=Au~temperature=x~wavelength=c':
            {'TEST1~material=Au~temperature=x~wavelength=0':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [0, 2]}),
            'TEST1~material=Au~temperature=x~wavelength=1':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [4, 6]})
            },
        'TEST1~material=Al~temperature=x~wavelength=c':
            {'TEST1~material=Al~temperature=x~wavelength=0':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [1, 3]}),
            'TEST1~material=Al~temperature=x~wavelength=1':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [5, 7]}),
            },
        'TEST1~material=c~temperature=x~wavelength=0':
            {'TEST1~material=Au~temperature=x~wavelength=0':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [0, 2]}),
            'TEST1~material=Al~temperature=x~wavelength=0':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [1, 3]})
            },
        'TEST1~material=c~temperature=x~wavelength=1':
            {'TEST1~material=Au~temperature=x~wavelength=1':
                pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [4, 6]}),
             'TEST1~material=Al~temperature=x~wavelength=1':
                 pd.DataFrame(
                        index=pd.Index([25.0, 35.0], name='temperature'),
                        data={'Value': [5, 7]}),
            },
        'TEST1~material=x~temperature=25.0~wavelength=c':
            {'TEST1~material=x~temperature=25.0~wavelength=0':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [1, 0]}),
            'TEST1~material=x~temperature=25.0~wavelength=1':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [5, 4]}),
            },

        'TEST1~material=x~temperature=35.0~wavelength=c':
            {'TEST1~material=x~temperature=35.0~wavelength=0':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [3, 2]}),
            'TEST1~material=x~temperature=35.0~wavelength=1':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [7, 6]}),
            },
        'TEST1~material=x~temperature=c~wavelength=0':
            {'TEST1~material=x~temperature=25.0~wavelength=0':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [1, 0]}),
            'TEST1~material=x~temperature=35.0~wavelength=0':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [3, 2]}),
            },
        'TEST1~material=x~temperature=c~wavelength=1':
            {'TEST1~material=x~temperature=25.0~wavelength=1':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [5, 4]}),
            'TEST1~material=x~temperature=35.0~wavelength=1':
                pd.DataFrame(
                        index=pd.Index(['Al', 'Au'], name='material'),
                        data={'Value': [7, 6]}),
            },
    }
    actual_data = exp.master_data_dict(data_3x2)
    assertDataDictEqual(actual_data, desired_data)
