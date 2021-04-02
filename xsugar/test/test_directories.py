import pytest
import numpy as np
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment

def testDirectoryAbsence(exp_data):
    """
    Tests that our test directories are properly created and destroyed
    """
    data_exists = os.path.isdir(exp_data['data_base_path'])
    figures_exist = os.path.isdir(exp_data['figures_base_path'])
    design_exist = os.path.isdir(exp_data['designs_base_path'])
    assert_equal(data_exists, False)
    assert_equal(figures_exist, False)
    assert_equal(design_exist, False)

def testDirectoryCreation(exp, exp_data):
    """
    Tests that we properly create the right directories and subdirectories
    """
    data_exists = os.path.isdir(exp_data['data_base_path'])
    figures_exist = os.path.isdir(exp_data['figures_base_path'])
    assert_equal(data_exists, True)
    assert_equal(figures_exist, True)

    data_exists = os.path.isdir(exp_data['data_base_path'] + 'TEST1/')
    figures_exist = os.path.isdir(exp_data['figures_base_path'] + 'TEST1/')
    assert_equal(data_exists, True)
    assert_equal(figures_exist, True)

def testDesignDirectoryCreation(exp, exp_data):
    """
    Tests that we properly create the right directories and subdirectories
    """
    exp = Experiment(name='TEST1', kind='design')
    data_exists = os.path.isdir(exp_data['designs_base_path'] + 'data/')
    figures_exist = os.path.isdir(exp_data['designs_base_path'] + 'figures/')
    assert_equal(data_exists, True)
    assert_equal(figures_exist, True)


def testNormalPath(exp, exp_data):
    """
    Tests that non-design paths resolve to the right locations with the
    format base/kind/name/.
    """
    desired_figures_path = exp_data['figures_full_path']
    desired_data_path = exp_data['data_full_path']
    actual_figures_path = exp.figures_full_path
    actual_data_path = exp.data_full_path
    assert_equal(actual_data_path, desired_data_path)
    assert_equal(actual_figures_path, desired_figures_path)

def testDesignPath(exp_data):
    exp = Experiment(name='TEST1', kind='design')
    actual_figures_path = exp.figures_full_path
    actual_data_path = exp.data_full_path
    desired_figures_path = exp_data['designs_base_path'] + 'figures/'
    desired_data_path = exp_data['designs_base_path'] + 'data/'

    assert_equal(actual_data_path, desired_data_path)
    assert_equal(actual_figures_path, desired_figures_path)
