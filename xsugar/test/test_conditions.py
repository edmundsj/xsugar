import pytest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment, factors_from_condition, get_partial_condition, condition_is_subset

def test_get_conditions(exp, convert_name):
    exp.data = {
        convert_name('TEST1~wavelength-1~temperature-25'): None,
        convert_name('TEST1~wavelength-2~temperature-25'): None,
        convert_name('TEST1~wavelength-3~temperature-25'): None,
        convert_name('TEST1~wavelength-1~temperature-35'): None,
        convert_name('TEST1~wavelength-2~temperature-35'): None,
        convert_name('TEST1~wavelength-3~temperature-35'): None,}
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

def test_get_conditions_exclude(exp, convert_name):
    exp.data = {
        convert_name('TEST1~wavelength-1~temperature-25'): None,
        convert_name('TEST1~wavelength-2~temperature-25'): None,
        convert_name('TEST1~wavelength-3~temperature-25'): None,
        convert_name('TEST1~wavelength-1~temperature-35'): None,
        convert_name('TEST1~wavelength-2~temperature-35'): None,
        convert_name('TEST1~wavelength-3~temperature-35'): None,}
    desired_conditions = [
        {'wavelength': 1},
        {'wavelength': 2},
        {'wavelength': 3},
    ]
    actual_conditions = exp.get_conditions(exclude='temperature')
    assert_equal(actual_conditions, desired_conditions)

def test_get_conditions_exclude_uneven(exp, convert_name):
    exp.data = {
        convert_name('TEST1~wavelength-1~temperature-25'): None,
        convert_name('TEST1~wavelength-2~temperature-25'): None,
        convert_name('TEST1~wavelength-3~temperature-25'): None,
        convert_name('TEST1~wavelength-1~temperature-35'): None,
        convert_name('TEST1~wavelength-2~temperature-35'): None,
        convert_name('TEST1~wavelength-3~temperature-35'): None,
        convert_name('TEST1~wavelength-4~temperature-35'): None,}
    desired_conditions = [
        {'wavelength': 1},
        {'wavelength': 2},
        {'wavelength': 3},
        {'wavelength': 4},
    ]
    actual_conditions = exp.get_conditions(exclude='temperature')
    assert_equal(actual_conditions, desired_conditions)

def test_factors_from_condition(exp):
    actual_factors = factors_from_condition(
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

def test_partial_condition_pass():
    cond = {'hi': 2, 'yo': 4, 'there': 12}
    desired_condition = cond
    actual_condition = get_partial_condition(cond)
    assert_equal(actual_condition, desired_condition)

def test_partial_condition_exclude():
    cond = {'hi': 2, 'yo': 4, 'there': 12}
    desired_condition = {'there': 12}
    actual_condition = get_partial_condition(
            cond, exclude_factors=['hi', 'yo'])
    assert_equal(actual_condition, desired_condition)

def test_partial_condition_include():
    cond= {'hi': 2, 'yo': 4, 'there': 12, 'swell': 4}
    desired_condition = {'there': 12}
    actual_condition = get_partial_condition(cond, include_factors='there')
    assert_equal(actual_condition, desired_condition)

def test_condition_is_subset_true():
    superset_cond = {'hi': 1, 'there': 2}
    subset_cond = {'hi': 1}
    is_subset_actual = condition_is_subset(subset_cond, superset_cond)
    is_subset_desired = True
    assert_equal(is_subset_actual, is_subset_desired)

def test_condition_is_subset_false():
    superset_cond = {'hi': 1, 'there': 2}
    subset_cond = {'hi': 1, 'there': 3}
    is_subset_actual = condition_is_subset(subset_cond, superset_cond)
    is_subset_desired = False
    assert_equal(is_subset_actual, is_subset_desired)

#def test_condition_from_name():
#name = 'TEST1~wavelength=25~
