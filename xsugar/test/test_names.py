"""
Tests data reading and writing operation, along with condition generation
"""
import pytest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment, ureg
from sciparse import assertDataDictEqual
from ast import literal_eval

def testNameFromCondition(exp, exp_data):
    """
    Tests whether we can properly generate a name from a specified
    condition
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
        'temperature' + ns + '25'
    condition = {'wavelength': 1, 'temperature': 25}
    filename_actual = exp.nameFromCondition(condition)
    assert_equal(filename_actual, filename_desired)

def testNameFromConditionWithID(exp, exp_data):
    """
    Tests whether we can properly generate a name from a specified
    condition
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    filename_desired = 'TEST1' + ns + 'id' + js +  'wavelength' + ns + \
                       '1' + js + 'temperature' + ns + '25'
    condition = {'ident': 'id', 'wavelength': 1, 'temperature': 25}
    filename_actual = exp.nameFromCondition(condition)
    assert_equal(filename_actual, filename_desired)

def test_name_from_condition_units(convert_name):
    sampling_frequency = 1.0 * ureg.Hz
    temperature = ureg.K * np.array([300, 305])
    wavelength = ureg.nm * np.array([500, 505])
    exp = Experiment(name='TEST1', kind='test',
            sampling_frequenc=sampling_frequency,
            wavelength=wavelength, temperature=temperature)
    actual_name = exp.nameFromCondition(exp.conditions[0])
    desired_name = convert_name('TEST1~wavelength=500nm~temperature=300K')
    assert_equal(actual_name, desired_name)

def testConditionFromName(exp, exp_data):
    """
    Tests whether we can generate a condition from a specified name
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
                       'temperature' + ns + '25'
    condition_desired = {'wavelength': 1, 'temperature': 25,
                         'frequency': 8500}
    condition_actual = exp.conditionFromName(filename_desired)
    assert_equal(condition_actual, condition_desired)

def testConditionFromNamePartial(exp, exp_data):
    """
    Tests whether we can generate a condition from a specified name
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    filename_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
                       'temperature' + ns + '25'
    condition_desired = {'wavelength': 1, 'temperature': 25}
    condition_actual = exp.conditionFromName(
        filename_desired, full_condition=False)
    assert_equal(condition_actual, condition_desired)

def test_condition_from_name_units(exp,exp_data, convert_name):
    name = convert_name('TEST1~wavelength=100nm~temperature=25K')
    desired_condition = {
        'wavelength': ureg.nm * 100,
        'temperature': ureg.degK * 25,
        'frequency': exp_data['frequency']}
    actual_condition = exp.conditionFromName(name)
    assertDataDictEqual(actual_condition, desired_condition)

def testConditionFromNameMetadata(exp, exp_data):
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    name_desired = 'TEST1' + js + 'wavelength' + ns + '1' + js + \
                       'temperature' + ns + '25'
    condition_desired = {'wavelength': 1, 'temperature': 25,
                         'frequency': 8500,
                         'horn': 'shoe'}
    exp.metadata[name_desired] = \
        {'horn': 'shoe'}

    condition_actual = exp.conditionFromName(name_desired)
    assert_equal(condition_actual, condition_desired)


def testConditionFromNameWithID(exp, exp_data):
    """
    Tests whether we can generate a condition from a specified name
    """
    js, ns = exp_data['major_separator'], exp_data['minor_separator']
    name_desired = 'TEST1' + ns + 'id' + js + 'wavelength' + ns + \
                       '1' + js + 'temperature' + ns + '25'
    condition_desired = {'ident': 'id', 'wavelength': 1,
                         'temperature': 25,
                         'frequency': 8500}
    condition_actual = exp.conditionFromName(name_desired)
    assert_equal(condition_actual, condition_desired)

def test_drop_name(exp, convert_name):
    initial_name = convert_name('TEST1~wavelength=0.01')
    desired_name = convert_name('wavelength=0.01')
    actual_name = exp.drop_name(initial_name)
    assert_equal(actual_name, desired_name)

def test_prettify_name(exp, convert_name):
    initial_name = convert_name('TEST1~wavelength=0.01~temperature=25.0')
    desired_name = 'wavelength=0.01, temperature=25.0'
    actual_name = exp.prettify_name(initial_name)
    assert_equal(actual_name, desired_name)
