"""
Tests data reading and writing operation, along with condition generation
"""
import unittest
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment, assertDataDictEqual
from ast import literal_eval
from itertools import zip_longest

class TestData(unittest.TestCase):


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


    def testExtractConstants(self):
        """
        Tests that our experiment properly extracts metadata from a list of
        arbitrary keyword arguments.
        """
        wavelengths = np.array([1, 2, 3])
        temperatures = np.array([25, 50])
        frequency = 8500
        exp = Experiment(name='TEST1', kind='test',
                         frequency=frequency, wavelengths=wavelengths,
                         temperatures=temperatures)
        actual_constants = exp.constants
        desired_constants = {'frequency': self.frequency}
        assert_equal(actual_constants , desired_constants)


    def test_extract_factors(self):
        """
        Tests proper extraction of conditions from arbitrary input keyword
        arguments.
        """
        actual_factors = self.exp.factors
        desired_factors = {'wavelengths': self.wavelengths,
                             'temperatures': self.temperatures}
        assert_equal(actual_factors, desired_factors)

    def test_generate_conditions(self):
        """
        Tests that our generator actually generates all the right combinations
        and in the right order.
        """
        expected_conds = [
            {'wavelengths': 1, 'temperatures': 25, 'frequency': 8500},
            {'wavelengths': 1, 'temperatures': 50, 'frequency': 8500},
            {'wavelengths': 2, 'temperatures': 25, 'frequency': 8500},
            {'wavelengths': 2, 'temperatures': 50, 'frequency': 8500},
            {'wavelengths': 3, 'temperatures': 25, 'frequency': 8500},
            {'wavelengths': 3, 'temperatures': 50, 'frequency': 8500},
        ]
        for actual_cond, desired_cond in  zip(self.exp.conditions,
                                              expected_conds):
            assert_equal(actual_cond, desired_cond)

    def testSaveRawResultsFilename(self):
        """
        Tests that we save the raw results in the proper directory and with
        the proper metadata and with the proper name.
        """
        raw_data = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Current': [4,4.5,6]})
        cond = {'wavelengths': 1, 'temperatures': 25, 'frequency': 8500}
        self.exp.saveRawResults(raw_data, cond)
        filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
        file_found = os.path.isfile(self.data_full_path + filename_desired)
        assert_equal(file_found, True)

    def testSaveRawResultsMetadata(self):
        """
        Tests that we save the raw results in the proper directory and with
        the proper metadata and with the proper name.
        """
        raw_data = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Current': [4,4.5,6]})
        cond = {'wavelengths': 1, 'temperatures': 25, 'frequency':
                self.frequency}
        self.exp.saveRawResults(raw_data, cond)
        filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
        with open(self.data_full_path + filename_desired) as fh:
            first_line = fh.readline()
            metadata_actual = literal_eval(first_line)

        metadata_desired = {'frequency': self.frequency}
        assert_equal(metadata_actual, metadata_desired)

    def testSaveRawResultsData(self):
        """
        Tests that we correctly save the raw data and can read it out again.
        """
        data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Current': [4,4.5,6]})
        cond = {'wavelengths': 1, 'temperatures': 25, 'frequency':
                self.frequency}
        self.exp.saveRawResults(data_desired, cond)
        filename_desired = 'TEST1~wavelengths-1~temperatures-25.csv'
        with open(self.data_full_path + filename_desired) as fh:
            first_line = fh.readline()
            data_actual = pd.read_csv(fh)
        assert_allclose(data_actual, data_desired)
        assert_equal(data_actual.columns.values, data_desired.columns.values)

    @pytest.mark.skip
    def testSaveDerivedQuantitiesFilename(self):
        master_data = {'TEST1': pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Mean': [1,2,4]})}
        cond = {'wavelengths': 1, 'temperatures': 25, 'frequency': 8500}
        self.exp.master_data = master_data
        self.exp.saveDerivedQuantities()
        filename_desired = 'TEST1.csv'
        file_found = os.path.isfile(self.data_full_path + filename_desired)
        assert_equal(file_found, True)

    @pytest.mark.skip
    def testSaveDerivedQuantitiesData(self):
        data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Mean': [1,2,4]})
        self.exp.master_data = data_desired
        self.exp.saveDerivedQuantities()
        filename_desired = 'TEST1.csv'
        with open(self.data_full_path + filename_desired) as fh:
            metadata = fh.readline()
            data_actual = pd.read_csv(fh)
        assert_allclose(data_actual, data_desired)
        assert_equal(data_actual.columns.values, data_desired.columns.values)

    @pytest.mark.skip
    def testSaveDerivedQuantitiesMetadata(self):
        data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Mean': [1,2,4]})
        metadata_desired = {'frequency': self.frequency}
        self.exp.master_data = data_desired
        self.exp.saveDerivedQuantities()
        filename_desired = 'TEST1.csv'
        with open(self.data_full_path + filename_desired) as fh:
            metadata_actual = literal_eval(fh.readline())
        assert_equal(metadata_actual, metadata_desired)

    def testLoadData(self):
        data_desired = pd.DataFrame({'wavelengths': [1, 2, 3],
                                 'Mean': [1,2,4]})
        metadata_desired = {'non': 'sense'}
        filename = 'TEST1~wavelengths-1~temperatures-25'
        file_extension = '.csv'
        full_filename = self.data_full_path + filename + file_extension
        with open(full_filename, 'w+') as fh:
            fh.write(str(metadata_desired) + '\n')
            data_desired.to_csv(fh, mode='a', index=False)
        self.exp.loadData()
        data_actual = self.exp.data[filename]
        assert_frame_equal(data_actual, data_desired)

    def testLoadXRDData(self):
        exp = Experiment(name='TEST1', kind='test_permanent')
        exp.loadXRDData()
        data_desired = pd.DataFrame({
            'Angle (deg)': [69.05, 69.055, 69.06, 69.065, 69.07,69.075,69.08,
            69.085, 69.09, 69.095, 69.1, 69.105, 69.11, 69.115],
            'Counts': [24, 30, 28, 40, 132, 272, 3472, 16368,21970,10562,
                       1210,264,130,64]})
        data_actual = exp.data['TEST1~1~type-locked_coupled~peak-Si']
        assert_frame_equal(data_actual, data_desired)

    def testLoadXRDMetadata(self):
        exp = Experiment(name='TEST1', kind='test_permanent')
        exp.loadXRDData()
        metadata_desired = {
            'date': '02/10/2021',
            'increment': 0.005, 'scantype': 'locked coupled',
            'start': 69.05, 'steps': 14, 'time': 1,
            'theta': 34.0, '2theta': 68.0, 'phi': 180.13, 'chi': -0.972}
        metadata_actual = exp.metadata['TEST1~1~type-locked_coupled~peak-Si']
        assert_equal(metadata_actual, metadata_desired)

    def testLoadConstants(self):
        """
        Tests that we can load metadata from a file successfully
        """
        wavelengths = np.array([1, 2, 3])
        temperatures = np.array([25, 50])
        frequency = 8500
        with open(self.data_full_path + \
                  'TEST1~wavelengths-1~temperatures-25.csv', 'w+') as fh:
            fh.write('{"frequency": 8500}\n')
            fh.write(f'Time, Data\n')
            fh.write(f'1, 2\n')
        exp = Experiment(name='TEST1', kind='test')
        exp.loadData()
        constants_actual = exp.constants
        constants_desired = {'frequency': frequency}
        assert_equal(constants_actual, constants_desired)

    def testLoadMetadata(self):
        """
        Tests that we can load metadata from a file successfully
        """
        wavelengths = np.array([1, 2, 3])
        temperatures = np.array([25, 50])
        frequency = 8500
        with open(self.data_full_path + \
                  'TEST1~wavelengths-1~temperatures-25.csv', 'w+') as fh:
            fh.write('{"frequency": 8500}\n')
            fh.write(f'Time, Data\n')
            fh.write(f'1, 2\n')
        exp = Experiment(name='TEST1', kind='test')
        exp.loadData()
        metadata_actual = exp.metadata
        metadata_desired = {'TEST1~wavelengths-1~temperatures-25': \
                            {'frequency': frequency}}
        assert_equal(metadata_actual, metadata_desired)

    # TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
    # THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

    # TODO: ADD TEST CASE TO ENSURE WE DON'T LOAD IN TOO MUCH DATA, OR DATA
    # THAT DOES NOT PRECISELY MATCH *BOTH* THE NAME *AND* THE ID.

    def testLookup(self):
        fudge_data = pd.DataFrame(
            {'Time (ms)': [1, 2, 3],
            'Voltage (V)': [0,0.1, 0.2]})
        self.exp.data = {'TEST1~wavelengths-1~temperatures-25':fudge_data,
                         'TEST1~wavelengths-2~temperatures-25':fudge_data,
                         'TEST1~wavelengths-2~temperatures-35':fudge_data,
                         'TEST1~wavelengths-2~temperatures-35':fudge_data,
                        }
        data_actual = self.exp.lookup(temperatures=25)
        data_desired = {'TEST1~wavelengths-1~temperatures-25':fudge_data,
                         'TEST1~wavelengths-2~temperatures-25':fudge_data,}
        assertDataDictEqual(data_actual, data_desired)

    def tearDown(self):
        """
        To be run after every test case. Removes directories we created if they
        exist.
        """
        rmtree(self.data_base_path, ignore_errors=True)
        rmtree(self.figures_base_path, ignore_errors=True)
        rmtree(self.designs_base_path, ignore_errors=True)

    @classmethod
    def tearDownClass(self):
        pass # Tear down to be run after entire script
