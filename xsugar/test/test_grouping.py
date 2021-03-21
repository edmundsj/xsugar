import pytest
from xsugar import Experiment, assertDataDictEqual
from pandas.testing import assert_frame_equal
from numpy.testing import assert_equal
import numpy as np
import pandas as pd
from itertools import zip_longest
from shutil import rmtree

@pytest.fixture
def exp():
    base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
    'pockels_modulator/'
    data_base_path = base_path + 'data/test/'
    data_full_path = base_path + 'data/test/TEST1/'
    figures_base_path = base_path + 'figures/test/'
    figures_full_path = base_path + 'figures/test/TEST1/'
    designs_base_path = base_path + 'designs/TEST1/'
    wavelengths = np.array([1, 2, 3])
    temperatures = np.array([25, 50])
    frequency = 8500
    fudge_data_1 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 1.3, 0.7]})
    fudge_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                               'Photocurrent (nA)': [0.5, 0.6, 0.7]})
    condition_1 = {'frequency': frequency,
                   'wavelength': 1,
                   'replicate': 0}
    condition_2 = {'frequency': frequency,
                   'wavelength': 1,
                   'replicate': 1}
    name_1 = 'TEST1~wavelength-1~replicate-0'
    name_2 = 'TEST1~wavelength-1~replicate-1'
    data_dict = {
        name_1: fudge_data_1,
        name_2: fudge_data_2}
    experiment = Experiment(name='TEST1', kind='test',
                     frequency=frequency,
                     wavelengths=wavelengths,
                     temperatures=temperatures)
    yield {
        'exp': experiment,
        'frequency': frequency,
        'wavelength': wavelengths,
        'temperature': temperatures,
        'name_1': name_1,
        'name_2': name_2,
        'fudge_data_1': fudge_data_1,
        'fudge_data_2': fudge_data_2,
        'condition_1': condition_1,
        'condition_2': condition_2,
        'data_dict': data_dict,}
    rmtree(data_base_path, ignore_errors=True)
    rmtree(figures_base_path, ignore_errors=True)
    rmtree(designs_base_path, ignore_errors=True)

def testGenerateGroupsNone(exp):
    desired_groups = exp['data_dict']
    actual_groups = exp['exp'].generate_groups(
        data_dict=exp['data_dict'], group_along=None)
    assertDataDictEqual(actual_groups, desired_groups)

def testGenerateGroups(exp):

    desired_groups = {'TEST1~wavelength-1': {
        'TEST1~wavelength-1~replicate-0': None,
        'TEST1~wavelength-1~replicate-1': None,
    }}

    actual_groups = exp['exp'].generate_groups(
        data_dict=exp['data_dict'], group_along='replicate')

    assertDataDictEqual(actual_groups, desired_groups)

def testGenerateGroupsWrongGroup(exp):
    """
    Confirms that we raise a valueError if we are not able to find the
    group
    """
    with pytest.raises(ValueError):
        exp['exp'].generate_groups(
                data_dict=exp['data_dict'], group_along='wrong_name')

def test_generate_groups_value(exp):
    desired_groups = {
        'TEST1~replicate-0': {
             'TEST1~wavelength-1~replicate-0': None},
        'TEST1~replicate-1': {
            'TEST1~wavelength-1~replicate-1': None}}
    actual_groups = exp['exp'].generate_groups(
            data_dict=exp['data_dict'], group_along='replicate', grouping_type='value')
    assertDataDictEqual(actual_groups, desired_groups)

def test_group_data_none(exp):
    """
    Assert that when group_along is none we just pass through the data
    """
    desired_data = exp['data_dict']
    actual_data = exp['exp'].group_data(
        data_dict=exp['data_dict'], group_along=None)
    assertDataDictEqual(actual_data, desired_data)

def test_group_scalar_data(exp):
    """
    Tests whether we can compress the derived quantity as a bunch of
    scalar-only dicts into a bunch of plottable and usaable pandas array.
    """
    scalar_data = {'TEST1~wavelength-1~temperature-25': 1,
                   'TEST1~wavelength-1~temperature-35': 2}

    desired_data = {'TEST1~wavelength-1': {
        'TEST1~wavelength-1~temperature-25': 1,
        'TEST1~wavelength-1~temperature-35': 2,}}

    actual_data = exp['exp'].group_data(
        data_dict=scalar_data, group_along='temperature')

    assertDataDictEqual(actual_data, desired_data)

def test_group_by_name(exp):

    actual_data = exp['exp'].group_data(
        data_dict=exp['data_dict'], group_along='replicate', grouping_type='name')
    desired_data = {
        'TEST1~wavelength-1': {
            'TEST1~wavelength-1~replicate-0': exp['fudge_data_1'],
            'TEST1~wavelength-1~replicate-1': exp['fudge_data_2']}}

    # Assert all values inside the pandas array are identical
    assertDataDictEqual(actual_data, desired_data)

def test_group_by_value(exp):
    actual_data = exp['exp'].group_data(
            data_dict=exp['data_dict'], group_along='wavelength', grouping_type='value')
    desired_data = {
        'TEST1~wavelength-1': {
            'TEST1~wavelength-1~replicate-0': exp['fudge_data_1'],
            'TEST1~wavelength-1~replicate-1': exp['fudge_data_2'],}}
    assertDataDictEqual(actual_data, desired_data)
