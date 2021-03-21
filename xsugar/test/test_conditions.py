import pytest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment

@pytest.fixture
def exp():
    base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
    'pockels_modulator/'
    data_base_path = base_path + 'data/test/'
    data_full_path = base_path + 'data/test/TEST1/'
    figures_base_path = base_path + 'figures/test/'
    figures_full_path = base_path + 'figures/test/TEST1/'
    designs_base_path = base_path + 'designs/TEST1/'
    wavelength = np.array([1, 2, 3])
    temperature = np.array([25, 50])
    frequency = 8500
    exp = Experiment(name='TEST1', kind='test',
                     frequency=frequency,
                     wavelength=wavelength,
                     temperature=temperature)
    yield exp
    rmtree(data_base_path, ignore_errors=True)
    rmtree(figures_base_path, ignore_errors=True)
    rmtree(designs_base_path, ignore_errors=True)

def test_get_conditions(exp):
    exp.data = {
        'TEST1~wavelength-1~temperature-25': None,
        'TEST1~wavelength-2~temperature-25': None,
        'TEST1~wavelength-3~temperature-25': None,
        'TEST1~wavelength-1~temperature-35': None,
        'TEST1~wavelength-2~temperature-35': None,
        'TEST1~wavelength-3~temperature-35': None,}
    desired_conditions = [
        {'wavelength': 1, 'temperature': 25},
        {'wavelength': 2, 'temperature': 25},
        {'wavelength': 3, 'temperature': 25},
        {'wavelength': 1, 'temperature': 35},
        {'wavelength': 2, 'temperature': 35},
        {'wavelength': 3, 'temperature': 35},
    ]
    actual_conditions = exp.get_conditions()
    assert_equal(actual_conditions, desired_conditions)

def test_get_conditions_exclude(exp):
    exp.data = {
        'TEST1~wavelength-1~temperature-25': None,
        'TEST1~wavelength-2~temperature-25': None,
        'TEST1~wavelength-3~temperature-25': None,
        'TEST1~wavelength-1~temperature-35': None,
        'TEST1~wavelength-2~temperature-35': None,
        'TEST1~wavelength-3~temperature-35': None,}
    desired_conditions = [
        {'wavelength': 1},
        {'wavelength': 2},
        {'wavelength': 3},
    ]
    actual_conditions = exp.get_conditions(exclude='temperature')
    assert_equal(actual_conditions, desired_conditions)

def test_get_conditions_exclude_uneven(exp):
    exp.data = {
        'TEST1~wavelength-1~temperature-25': None,
        'TEST1~wavelength-2~temperature-25': None,
        'TEST1~wavelength-3~temperature-25': None,
        'TEST1~wavelength-1~temperature-35': None,
        'TEST1~wavelength-2~temperature-35': None,
        'TEST1~wavelength-3~temperature-35': None,
        'TEST1~wavelength-4~temperature-35': None,}
    desired_conditions = [
        {'wavelength': 1},
        {'wavelength': 2},
        {'wavelength': 3},
        {'wavelength': 4},
    ]
    actual_conditions = exp.get_conditions(exclude='temperature')
    assert_equal(actual_conditions, desired_conditions)

def test_factors_from_condition(exp):
    actual_factors = exp.factors_from_condition(
            {'wavelength': 2, 'temperature': 5, 'material': 'Al'})
    desired_factors = ['wavelength', 'temperature', 'material']
    assert_equal(actual_factors, desired_factors)

def test_append_condition(exp):
    desired_conditions = [
        {'wavelength': 1, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 1, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 5, 'temperature': 0, 'glue': 9, 'frequency': 8500}
    ]
    exp.append_condition(wavelength=5, temperature=0, glue=9)
    actual_conditions = exp.conditions
    assert_equal(actual_conditions, desired_conditions)

def test_insert_condition(exp):
    desired_conditions = [
        {'wavelength': 1, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 1, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 5, 'temperature': 0, 'glue': 9, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 50, 'frequency': 8500},
    ]
    exp.insert_condition(2, wavelength=5, temperature=0, glue=9)
    actual_conditions = exp.conditions
    assert_equal(actual_conditions, desired_conditions)

def test_generate_conditions_1to1(exp):
    desired_conditions = [
        {'wavelength': 1, 'temperature': 25, 'frequency': 8500},
        {'wavelength': 2, 'temperature': 50, 'frequency': 8500},
        {'wavelength': 3, 'temperature': 75, 'frequency': 8500},
    ]
    actual_conditions = exp.generate_conditions(
            comb_type='1to1',
            wavelength=[1, 2, 3], temperature=[25,50,75])
    assert_equal(actual_conditions, desired_conditions)

