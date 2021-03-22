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

def testSavePSDFigureFilename(exp, path_data, convert_name):
    """
    Tests that we successfully created and saved a single PSD figure when
    our dataset is just a single item.
    """
    raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    condition_name = convert_name('TEST1~wavelengths-1~temperatures-25')
    filename_desired = condition_name + '~PSD.png'
    exp.data = {condition_name: raw_data}
    exp.plotPSD(average_along=None)
    file_found = os.path.isfile(path_data['figures_full_path'] + filename_desired)
    assert_equal(file_found, True)

def testSavePSDFigureMultipleFilename(exp, path_data, convert_name):
    """
    Tests that we successfully created and saved a single PSD figure when
    our dataset is just a single item.
    """
    raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    condition_name_1 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-0')
    condition_name_2 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-1')
    filename_desired_1 = condition_name_1 + '~PSD.png'
    filename_desired_2 = condition_name_2 + '~PSD.png'
    exp.data = {condition_name_1: raw_data, condition_name_2: raw_data}
    exp.plotPSD(average_along=None)
    file1_found = os.path.isfile(path_data['figures_full_path'] +
                                 filename_desired_1)
    file2_found = os.path.isfile(path_data['figures_full_path'] +
                                 filename_desired_2)
    assert_equal(file1_found, True)
    assert_equal(file2_found, True)

def testSavePSDFigureAverageFilename(exp, path_data, convert_name):
    """
    Tests that we successfully created and saved a single PSD figure when
    we want to create an averaged PSD plot
    """
    raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [4,4.5,6]})
    raw_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [8,4.5,8]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    condition_name_1 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-1')
    condition_name_2 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-2')
    filename_desired = convert_name('TEST1~wavelengths-1~temperatures-25~PSD~averaged.png')

    exp.data = {condition_name_1: raw_data,
                 condition_name_2: raw_data_2}
    exp.plotPSD(average_along='replicate')
    file_found = os.path.isfile(path_data['figures_full_path'] + filename_desired)
    assert_equal(file_found, True)

def testGenerateTimeDomainPlot(exp, path_data, convert_name):
    """
    Tests that we successfully create a simple figure from a single pandas
    array.
    """
    raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [4,4.5,6]})
    cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
    condition_name = convert_name('TEST1~wavelengths-1~temperatures-25_replicate-1')
    filename_desired = condition_name + '.png'
    exp.data = {condition_name: raw_data}
    exp.plot()
    file_found = os.path.isfile(path_data['figures_full_path'] + filename_desired)
    assert_equal(file_found, True)

def testGenerateRepresentativePlot(exp, path_data, convert_name):
    """
    Tests that we successfully create a single figure from a whole set of
    replicate data, instead of a bunch of figures
    """
    raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                             'Current (mV)': [4,4.5,6]})
    cond_1 = {'wavelength': 1, 'temperature': 25, 'frequency': 8500,
              'replicate': 1}
    cond_2 = {'wavelength': 1, 'temperature': 25, 'frequency': 8500,
              'replicate': 2}
    condition_name_1 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-1')
    condition_name_2 = convert_name('TEST1~wavelengths-1~temperatures-25~replicate-2')

    filename_desired = convert_name('TEST1~wavelengths-1~temperatures-25~representative')
    exp.data = {condition_name_1: raw_data,
                 condition_name_2 : raw_data}
    exp.plot(representative='replicate')
    files_found = os.listdir(path_data['figures_full_path'])
    file_1_found = \
        os.path.isfile(path_data['figures_full_path'] + filename_desired + '.png')
    assert_equal(file_1_found, True)
    assert_equal(len(files_found), 1)
