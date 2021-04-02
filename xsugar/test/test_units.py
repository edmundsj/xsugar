import pint
import pytest
from xsugar import Experiment
from sciparse import assertDataDictEqual, assert_equal_qt
from pint import UnitRegistry
from numpy.testing import assert_equal
import numpy as np


@pytest.fixture
def exp(ureg):
    wavelength = ureg.nm * np.array([500, 505])
    temperature = ureg.degK * np.array([300, 305])
    my_exp = Experiment(kind='test', name='TEST1')
    return my_exp

def test_units_init(ureg):
    wavelength = ureg.nm * np.array([500, 505])
    temperature = ureg.degK * np.array([25.0, 35.0])
    sampling_frequency = ureg.Hz * 1.0
    my_exp = Experiment(kind='test', name='TEST1',
            wavelength=wavelength, temperature=temperature,
            sampling_frequency=sampling_frequency)
    desired_factors = {
        'wavelength': wavelength,
        'temperature': temperature}
    desired_constants = {'sampling_frequency': sampling_frequency}
    actual_factors = my_exp.factors
    actual_constants = my_exp.constants
    assertDataDictEqual(actual_factors, desired_factors)
    assertDataDictEqual(actual_constants, desired_constants)
