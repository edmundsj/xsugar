import unittest
import numpy as np
import os
from shutil import rmtree
from numpy.testing import assert_equal, assert_allclose
from xsugar import Experiment

class TestDirectories(unittest.TestCase):
    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
        'pockels_modulator/'
        cls.data_base_path = cls.base_path + 'data/test/'
        cls.figures_base_path = cls.base_path + 'figures/test/'
        cls.designs_base_path = cls.base_path + 'designs/TEST1/'

    def testDirectoryAbsence(self):
        """
        Tests that our test directories are properly created and destroyed
        """
        data_exists = os.path.isdir(self.data_base_path)
        figures_exist = os.path.isdir(self.figures_base_path)
        design_exist = os.path.isdir(self.designs_base_path)
        assert_equal(data_exists, False)
        assert_equal(figures_exist, False)
        assert_equal(design_exist, False)

    def testDirectoryCreation(self):
        """
        Tests that we properly create the right directories and subdirectories
        """
        exp = Experiment(name='TEST1', kind='test')
        data_exists = os.path.isdir(self.data_base_path)
        figures_exist = os.path.isdir(self.figures_base_path)
        assert_equal(data_exists, True)
        assert_equal(figures_exist, True)

        data_exists = os.path.isdir(self.data_base_path + 'TEST1/')
        figures_exist = os.path.isdir(self.figures_base_path + 'TEST1/')
        assert_equal(data_exists, True)
        assert_equal(figures_exist, True)

    def testDesignDirectoryCreation(self):
        """
        Tests that we properly create the right directories and subdirectories
        """
        exp = Experiment(name='TEST1', kind='design')
        data_exists = os.path.isdir(self.designs_base_path + 'data/')
        figures_exist = os.path.isdir(self.designs_base_path + 'figures/')
        assert_equal(data_exists, True)
        assert_equal(figures_exist, True)


    def testNormalPath(self):
        """
        Tests that non-design paths resolve to the right locations with the
        format base/kind/name/.
        """
        exp = Experiment(name='TEST1', kind='test')
        desired_figures_path = self.figures_base_path + 'TEST1/'
        desired_data_path = self.data_base_path + 'TEST1/'
        actual_figures_path = exp.figures_full_path
        actual_data_path = exp.data_full_path
        assert_equal(actual_data_path, desired_data_path)
        assert_equal(actual_figures_path, desired_figures_path)

    def testDesignPath(self):
        exp = Experiment(name='TEST1', kind='design')
        actual_figures_path = exp.figures_full_path
        actual_data_path = exp.data_full_path
        desired_figures_path = self.designs_base_path + 'figures/'
        desired_data_path = self.designs_base_path + 'data/'

        assert_equal(actual_data_path, desired_data_path)
        assert_equal(actual_figures_path, desired_figures_path)

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
