import pytest
from xsugar import ureg, Experiment, match_theory_data
from sciparse import assert_equal_qt, assert_allclose_qt, assertDataDictEqual
from numpy.testing import assert_equal
from sugarplot import assert_figures_equal, plt, Figure
import pandas as pd
import numpy as np
import os

@pytest.fixture
def sim_exp():
    sim_exp =  Experiment(name='REFL2', kind='test')
    sim_exp.data = {
        'REFL2~material=Au~spectra=R0': 0,
        'REFL2~material=Al~spectra=R0': 1,
        'REFL2~material=AlN~spectra=R0': 2,
        'REFL2~material=AlN~modulation_voltage=5V~spectra=dR': 3,
        'REFL2~material=AlN~modulation_voltage=10V~spectra=dR': 4
    }
    sim_exp.conditions = [
        {'material': 'Au', 'spectra': 'R0'},
        {'material': 'Al', 'spectra': 'R0'},
        {'material': 'AlN', 'spectra': 'R0'},
        {'material': 'AlN', 'spectra': 'dR', 'modulation_voltage': 5*ureg.V},
        {'material': 'AlN', 'spectra': 'dR', 'modulation_voltage': 10*ureg.V},
    ]
    return sim_exp

def test_match_theory_data_modulation(sim_exp, convert_name):
    curve_name = convert_name('TEST1~material=AlN~modulation_voltage=10V~spectra=dR~wavelength=x')
    matched_data_actual = match_theory_data(curve_name, sim_exp)
    matched_data_desired = 4
    assert_equal(matched_data_actual, matched_data_desired)

def test_match_theory_data_nomatch(sim_exp, convert_name):
    curve_name = convert_name('TEST1~material=AlN~modulation_voltage=20V~spectra=dR~wavelength=x')
    matched_data_actual = match_theory_data(curve_name, sim_exp)
    matched_data_desired = None
    assert_equal(matched_data_actual, matched_data_desired)
