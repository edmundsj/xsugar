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
from xsugar import assertDataDictEqual

class TestExecution(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.array([1, 2])
        self.temperature = np.array([25, 50])
        self.replicate = np.array([0, 1])
        self.frequency = 8500
        self.fake_data = pd.DataFrame(
            {'Time (ms)': [0, 0.1, 0.2],
             'Voltage (V)': [1, 2, 3]})
        def dataFunc(cond):
            fake_data = pd.DataFrame(
                {'Time (ms)': [0, 0.1, 0.2],
                 'Voltage (V)': [1, 2, 3]})
            return fake_data

        self.exp = Experiment(
            name='TEST1', kind='test', measure_func=dataFunc,
             frequency=self.frequency, wavelength=self.wavelength,
             temperature=self.temperature, replicate=self.replicate)

    @classmethod
    def setUpClass(cls):
        cls.base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
        'pockels_modulator/'
        cls.data_base_path = cls.base_path + 'data/test/'
        cls.data_full_path = cls.base_path + 'data/test/TEST1/'
        cls.figures_base_path = cls.base_path + 'figures/test/'
        cls.figures_full_path = cls.base_path + 'figures/test/TEST1/'
        cls.designs_base_path = cls.base_path + 'designs/TEST1/'

    def testExecuteExperimentFilesWritten(self):
        """
        Tests that we successfully load a dataset
        """
        files = [
            'TEST1~wavelength-1~temperature-25~replicate-0',
            'TEST1~wavelength-1~temperature-25~replicate-1',
            'TEST1~wavelength-1~temperature-50~replicate-0',
            'TEST1~wavelength-1~temperature-50~replicate-1',
            'TEST1~wavelength-2~temperature-25~replicate-0',
            'TEST1~wavelength-2~temperature-25~replicate-1',
            'TEST1~wavelength-2~temperature-50~replicate-0',
            'TEST1~wavelength-2~temperature-50~replicate-1']
        full_filenames = [self.data_full_path + fn + '.csv' for fn in files]
        self.exp.Execute()
        files_found = [os.path.isfile(fn) for fn in full_filenames]
        assert_equal(all(files_found), True)

    def testExecuteExperimentDataCorrect(self):
        self.exp.Execute()
        desired_data = {
            'TEST1~wavelength-1~temperature-25~replicate-0': self.fake_data,
            'TEST1~wavelength-1~temperature-25~replicate-1': self.fake_data,
            'TEST1~wavelength-1~temperature-50~replicate-0': self.fake_data,
            'TEST1~wavelength-1~temperature-50~replicate-1': self.fake_data,
            'TEST1~wavelength-2~temperature-25~replicate-0': self.fake_data,
            'TEST1~wavelength-2~temperature-25~replicate-1': self.fake_data,
            'TEST1~wavelength-2~temperature-50~replicate-0': self.fake_data,
            'TEST1~wavelength-2~temperature-50~replicate-1': self.fake_data}
        actual_data = self.exp.data
        assertDataDictEqual(actual_data, desired_data)

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
