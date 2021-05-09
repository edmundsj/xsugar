from xsugar import Experiment
import pytest
from numpy.testing import assert_equal
from shutil import rmtree

@pytest.fixture
def exp(exp_data, ureg):
    exp = Experiment(name='TEST1', kind='test',
                     frequency=exp_data['frequency'],
                     wavelength=[1, 2],
                     temperature=[25, 35])
    yield exp
    rmtree(exp_data['data_base_path'], ignore_errors=True)
    rmtree(exp_data['figures_base_path'], ignore_errors=True)
    rmtree(exp_data['designs_base_path'], ignore_errors=True)

def test_setup_conditions(exp, exp_data):
    """
    Check to see that given a cartesian product of conditions that percolates appropriately
    """
    actual_conditions = exp.conditions
    desired_conditions = [
        {'wavelength': 1, 'temperature': 25,
            'frequency': exp_data['frequency']},
        {'wavelength': 1, 'temperature': 35,
            'frequency': exp_data['frequency']},
        {'wavelength': 2, 'temperature': 25,
            'frequency': exp_data['frequency']},
        {'wavelength': 2, 'temperature': 35,
            'frequency': exp_data['frequency']},
    ]
    assert_equal(actual_conditions, desired_conditions)
