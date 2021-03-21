import pytest
import numpy as np
from shutil import rmtree
from xsugar import Experiment
from pathlib import Path
import os

@pytest.fixture
def exp(exp_data):
    exp = Experiment(name='TEST1', kind='test',
                     frequency=exp_data['frequency'],
                     wavelength=exp_data['wavelength'],
                     temperature=exp_data['temperature'])
    yield exp
    rmtree(exp_data['data_base_path'], ignore_errors=True)
    rmtree(exp_data['figures_base_path'], ignore_errors=True)
    rmtree(exp_data['designs_base_path'], ignore_errors=True)

@pytest.fixture
def path_data():
    base_path = Path.home()
    if 'LOGNAME' in os.environ:
        if os.environ['LOGNAME'] == 'jordan.e':
            base_path = '/Users/jordan.e/Google Drive/GTD/PhD/docs/' +\
                'pockels_modulator/'
    data_base_path = base_path + 'data/test/'
    data_full_path = base_path + 'data/test/TEST1/'
    figures_base_path = base_path + 'figures/test/'
    figures_full_path = base_path + 'figures/test/TEST1/'
    designs_base_path = base_path + 'designs/TEST1/'
    yield {
        'data_base_path': data_base_path,
        'data_full_path': data_full_path,
        'figures_base_path': figures_base_path,
        'figures_full_path': figures_full_path,
        'designs_base_path': designs_base_path,
    }

@pytest.fixture
def exp_data(path_data):
    wavelength = np.array([1, 2, 3])
    temperature = np.array([25, 50])
    frequency = 8500
    yield dict({
        'wavelength': wavelength,
        'temperature': temperature,
        'frequency': frequency,
    }, **path_data)
