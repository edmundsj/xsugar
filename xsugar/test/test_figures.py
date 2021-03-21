"""
Tests data reading and writing operation, along with condition generation
"""
import unittest
import numpy as np
import pandas as pd
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment
from ast import literal_eval

class TestFigures(unittest.TestCase):
    def setUp(self):
        self.wavelengths = np.array([1, 2, 3])
        self.temperatures = np.array([25, 50])
        self.frequency = 8500
        self.exp = Experiment(name='TEST1', kind='test',
                         frequency=self.frequency,
                         wavelengths=self.wavelengths,
                         temperatures=self.temperatures)

    @classmethod
    def setUpClass(cls):
        cls.base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
        'pockels_modulator/'
        cls.data_base_path = cls.base_path + 'data/test/'
        cls.data_full_path = cls.base_path + 'data/test/TEST1/'
        cls.figures_base_path = cls.base_path + 'figures/test/'
        cls.figures_full_path = cls.base_path + 'figures/test/TEST1/'
        cls.designs_base_path = cls.base_path + 'designs/TEST1/'


    def testSavePSDFigureFilename(self):
        """
        Tests that we successfully created and saved a single PSD figure when
        our dataset is just a single item.
        """
        raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                                 'Current (mV)': [4,4.5,6]})
        cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
        condition_name = 'TEST1~wavelengths-1~temperatures-25'
        filename_desired = condition_name + '~PSD.png'
        self.exp.data = {condition_name: raw_data}
        self.exp.plotPSD(average_along=None)
        file_found = os.path.isfile(self.figures_full_path + filename_desired)
        assert_equal(file_found, True)

    def testSavePSDFigureMultipleFilename(self):
        """
        Tests that we successfully created and saved a single PSD figure when
        our dataset is just a single item.
        """
        raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                                 'Current (mV)': [4,4.5,6]})
        cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
        condition_name_1 = 'TEST1~wavelengths-1~temperatures-25~replicate-0'
        condition_name_2 = 'TEST1~wavelengths-1~temperatures-25~replicate-1'
        filename_desired_1 = condition_name_1 + '~PSD.png'
        filename_desired_2 = condition_name_2 + '~PSD.png'
        self.exp.data = {condition_name_1: raw_data, condition_name_2: raw_data}
        self.exp.plotPSD(average_along=None)
        file1_found = os.path.isfile(self.figures_full_path +
                                     filename_desired_1)
        file2_found = os.path.isfile(self.figures_full_path +
                                     filename_desired_2)
        assert_equal(file1_found, True)
        assert_equal(file2_found, True)

    def testSavePSDFigureAverageFilename(self):
        """
        Tests that we successfully created and saved a single PSD figure when
        we want to create an averaged PSD plot
        """
        raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                                 'Current (mV)': [4,4.5,6]})
        raw_data_2 = pd.DataFrame({'Time (ms)': [1, 2, 3],
                                 'Current (mV)': [8,4.5,8]})
        cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
        condition_name_1 = 'TEST1~wavelengths-1~temperatures-25~replicate-1'
        condition_name_2 = 'TEST1~wavelengths-1~temperatures-25~replicate-2'
        filename_desired = 'TEST1~wavelengths-1~temperatures-25~PSD~averaged.png'

        self.exp.data = {condition_name_1: raw_data,
                     condition_name_2: raw_data_2}
        self.exp.plotPSD(average_along='replicate')
        file_found = os.path.isfile(self.figures_full_path + filename_desired)
        assert_equal(file_found, True)

    def testGenerateTimeDomainPlot(self):
        """
        Tests that we successfully create a simple figure from a single pandas
        array.
        """
        raw_data = pd.DataFrame({'Time (ms)': [1, 2, 3],
                                 'Current (mV)': [4,4.5,6]})
        cond = {'wavelength': 1, 'temperature': 25, 'frequency': 8500}
        condition_name = 'TEST1~wavelengths-1~temperatures-25_replicate-1'
        filename_desired = condition_name + '.png'
        self.exp.data = {condition_name: raw_data}
        self.exp.plot()
        file_found = os.path.isfile(self.figures_full_path + filename_desired)
        assert_equal(file_found, True)

    def testGenerateRepresentativePlot(self):
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
        condition_name_1 = 'TEST1~wavelengths-1~temperatures-25~replicate-1'
        condition_name_2 = 'TEST1~wavelengths-1~temperatures-25~replicate-2'

        filename_desired = 'TEST1~wavelengths-1~temperatures-25~representative'
        self.exp.data = {condition_name_1: raw_data,
                     condition_name_2 : raw_data}
        self.exp.plot(representative='replicate')
        files_found = os.listdir(self.figures_full_path)
        file_1_found = \
            os.path.isfile(self.figures_full_path + filename_desired + '.png')
        assert_equal(file_1_found, True)
        assert_equal(len(files_found), 1)


    def tearDown(self):
        """
        co be run after every test case. Removes directories we created if they
        exist.
        """
        rmtree(self.data_base_path, ignore_errors=True)
        rmtree(self.figures_base_path, ignore_errors=True)
        rmtree(self.designs_base_path, ignore_errors=True)

    @classmethod
    def tearDownClass(self):
        pass # Tear down to be run after entire script
