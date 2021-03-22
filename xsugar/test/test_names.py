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

