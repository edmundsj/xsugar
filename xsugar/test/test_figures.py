"""
Tests data reading and writing operation, along with condition generation
"""
import pytest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from matplotlib.figure import Figure
from xsugar import Experiment, ureg
from sugarplot import assert_figures_equal, prettifyPlot
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

def test_generate_plot_1var(exp, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1',
        'TEST1~wavelength=2',
        ]]
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
    }
    actual_figs, actual_axes = exp.plot(data_dict)
    actual_fig = actual_figs[0]
    actual_ax = actual_axes[0]
    desired_fig = Figure()
    desired_ax = desired_fig.subplots(subplot_kw={'xlabel': 'wavelength', 'ylabel': 'Value'})
    desired_ax.plot([1, 2], [1.0, 2.0])
    prettifyPlot(fig=desired_fig, ax=desired_ax)
    assert_figures_equal(actual_fig, desired_fig)

def test_generate_plot_1var_units(exp_units, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1nm',
        'TEST1~wavelength=2nm',
        ]]
    data_dict = {
        names[0]: 1.0 * ureg.nA,
        names[1]: 2.0 * ureg.nA,
    }
    actual_figs, actual_axes = exp_units.plot(data_dict)
    actual_fig = actual_figs[0]
    actual_ax = actual_axes[0]
    desired_fig = Figure()
    desired_ax = desired_fig.subplots(subplot_kw={'xlabel': 'wavelength (nm)', 'ylabel': 'current (nA)'})
    desired_ax.plot([1, 2], [1.0, 2.0])
    prettifyPlot(fig=desired_fig, ax=desired_ax)
    assert_figures_equal(actual_fig, desired_fig)

def test_generate_plot_2var(exp, convert_name):
    names = [convert_name(name) for name in \
        [
        'TEST1~wavelength=1~temperature=25.0',
        'TEST1~wavelength=2~temperature=25.0',
        'TEST1~wavelength=1~temperature=35.0',
        'TEST1~wavelength=2~temperature=35.0',
        ]]
    data_dict = {
        names[0]: 1.0,
        names[1]: 2.0,
        names[2]: 3.0,
        names[3]: 4.0,
    }
    actual_figs, actual_axes = exp.plot(data_dict)
    assert_equal(len(actual_figs), 2)
    assert_equal(len(actual_axes), 2)

    desired_fig0 = Figure()
    desired_ax0 = desired_fig0.subplots(subplot_kw={'xlabel': 'wavelength', 'ylabel': 'Value'})
    desired_ax0.plot([1, 2], [1.0, 2.0])
    desired_ax0.plot([1, 2], [3.0, 4.0])
    prettifyPlot(fig=desired_fig0, ax=desired_ax0)

    desired_fig1 = Figure()
    desired_ax1 = desired_fig1.subplots(subplot_kw={'xlabel': 'temperature', 'ylabel': 'Value'})
    desired_ax1.plot([25.0, 35.0], [1.0, 3.0])
    desired_ax1.plot([25.0, 35.0], [2.0, 4.0])
    prettifyPlot(fig=desired_fig1, ax=desired_ax1)

    assert_figures_equal(actual_figs[0], desired_fig0)
    assert_figures_equal(actual_figs[1], desired_fig1)
