import pytest
from xsugar import Experiment
from sciparse import assertDataDictEqual
from pandas.testing import assert_frame_equal
from numpy.testing import assert_equal
import numpy as np
import pandas as pd
from itertools import zip_longest
from shutil import rmtree

@pytest.fixture
def wav_temp_data():
    ind_vals = ((1, 25), (2, 25), (1, 35), (2, 35))
    ind = pd.MultiIndex.from_tuples(ind_vals,
            names=('wavelength', 'temperature'))
    data = pd.DataFrame(index=ind, data={'current (nA)': [0, 1, 2, 3]})
    yield data

@pytest.fixture
def exp(path_data, convert_name):
    frequency = 8500
    experiment = Experiment(name='TEST1', kind='test',
                     frequency=frequency)
    yield experiment

    rmtree(path_data['data_base_path'], ignore_errors=True)
    rmtree(path_data['figures_base_path'], ignore_errors=True)
    rmtree(path_data['designs_base_path'], ignore_errors=True)

def test_generate_groups_passthrough(wav_temp_data, exp):
    actual_groups = wav_temp_data
    exp.group_data(data=wav_temp_data, group_along=None)
    desired_groups = wav_temp_data
    assert_frame_equal(actual_groups, desired_groups)

def test_group_data_value_2x2(exp, wav_temp_data, convert_name):

    desired_groups = {
        convert_name('TEST1~wavelength=1'): wav_temp_data.iloc[[0, 2]],
        convert_name('TEST1~wavelength=2'): wav_temp_data.iloc[[1, 3]],
        }

    actual_groups = exp.group_data(
        data=wav_temp_data, group_along='wavelength',
        grouping_type='value')

    assertDataDictEqual(actual_groups, desired_groups)

def test_group_data_name_2x2(exp, wav_temp_data, convert_name):

    desired_groups = {
        convert_name('TEST1~wavelength=1'): wav_temp_data.iloc[[0, 2]],
        convert_name('TEST1~wavelength=2'): wav_temp_data.iloc[[1, 3]],
        }

    actual_groups = exp.group_data(
        data=wav_temp_data, group_along='temperature',
        grouping_type='name')

    assertDataDictEqual(actual_groups, desired_groups)

def testGenerateGroupsWrongGroup(exp, wav_temp_data):
    """
    Confirms that we raise a valueError if we are not able to find the
    group
    """
    with pytest.raises(ValueError):
        exp.group_data(data=wav_temp_data, group_along='wrong_name')
