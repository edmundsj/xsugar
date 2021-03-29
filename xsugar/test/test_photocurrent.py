import pytest
from xsugar import ureg, dc_photocurrent, modulated_photocurrent, noise_current, Experiment, assertDataDictEqual
from sciparse import assert_equal_qt, assert_allclose_qt
import pandas as pd
import numpy as np

def test_dc_photocurrent():
    gain = 10 * ureg.Mohm
    input_data = pd.DataFrame({
            'time (ms)': [0, 0.01, 0.02],
            'voltage (uV)': [0.1, 0.2, 0.3]})
    desired_photocurrent = 2e-5*ureg.nA
    cond = {'gain': gain}
    actual_photocurrent = dc_photocurrent(input_data, cond)
    assert_allclose_qt(actual_photocurrent, desired_photocurrent)

def test_modulated_photocurrent():
    data_length = 1000
    sampling_frequency = 1 * ureg.kHz
    signal_frequency = 0.1 * ureg.kHz
    sampling_period = (1 / sampling_frequency).to(ureg.s)
    gain = 1.5 * ureg.Mohm
    cond = {'gain': gain, 'sampling_frequency': 1 * ureg.kHz}

    number_periods = int(np.floor(data_length / (sampling_frequency / signal_frequency)))
    number_sync_points = number_periods
    indices = np.arange(0, number_sync_points, 1)
    sync_indices = (1/2* sampling_frequency / signal_frequency *(1 + 2*indices)).astype(np.int).magnitude
    zero_column = np.zeros(data_length, dtype=np.int)
    zero_column[sync_indices] = 1

    times = np.arange(0, data_length, 1)*sampling_period
    voltages = 0.5 * ureg.mV * np.sin(2*np.pi * signal_frequency * times)
    data = pd.DataFrame({
            'time (ms)': times.to(ureg.ms).magnitude,
            'voltage (mV)': voltages.to(ureg.mV).magnitude,
            'Sync': zero_column
            })
    desired_data = (0.5 * ureg.mV / gain).to(ureg.pA) / np.sqrt(2)
    actual_data = modulated_photocurrent(data, cond)
    assert_allclose_qt(actual_data, desired_data)

def test_noise_current():
    data_length = 10000
    sampling_frequency = 10*ureg.kHz
    nyquist_frequency = sampling_frequency / 2
    sampling_period = (1 / sampling_frequency).to(ureg.s)
    noise_voltage = np.random.normal(size=data_length)*ureg.uV
    filter_cutoff = 200*ureg.Hz
    times = np.arange(0, data_length, 1)*sampling_period
    data = pd.DataFrame({
            'Time (ms)': times.to(ureg.ms).magnitude,
            'voltage (uV)': noise_voltage.to(ureg.uV).magnitude,
            })
    gain = 1.0 * ureg.Mohm
    cond = {
        'gain': gain,
        'filter_cutoff': filter_cutoff
    }
    filter_correction_factor = (nyquist_frequency - filter_cutoff) / \
                               nyquist_frequency
    desired_noise_current_psd = \
         np.square(noise_voltage.std() / gain) / nyquist_frequency
    desired_noise_current_psd = desired_noise_current_psd.to(
            ureg.A ** 2 / ureg.Hz)
    actual_noise_current_psd = noise_current(data, cond)
    assert_allclose_qt(
            actual_noise_current_psd, desired_noise_current_psd,
            atol=1e-31, rtol=1e-2)

def test_noise_current_bin2():
    data_length = 3000
    sampling_frequency = 10*ureg.kHz
    sampling_period = (1 / sampling_frequency).to(ureg.s)
    noise_voltage = np.random.normal(size=data_length)*ureg.uV
    times = np.arange(0, data_length, 1)*sampling_period
    data = pd.DataFrame({
            'Time (ms)': times.to(ureg.ms).magnitude,
            'voltage (uV)': noise_voltage.to(ureg.uV).magnitude,
            })
    gain = 1.0 * ureg.Mohm
    cond = {
        'gain': gain,
        'filter_cutoff': 200*ureg.Hz
    }
    desired_noise_current_psd = \
         np.square(noise_voltage.std() / gain) / (sampling_frequency / 2)
    desired_noise_current_psd = desired_noise_current_psd.to(
            ureg.A ** 2 / ureg.Hz)
    actual_noise_current_psd = noise_current(data, cond)
    assert_allclose_qt(
            actual_noise_current_psd, desired_noise_current_psd,
            atol=1e-31, rtol=1e-2)

def test_process_photocurrent_simple(convert_name):
    """
    Verifies that, given a sinusoidal input with a known offset and amplitude, the correct data is generated.
    """
    wavelength = np.array([700, 750]) * ureg.nm
    material = ['Au', 'Al']
    gain = 1 * ureg.Mohm
    current_offset = 100
    current_amplitude = 1
    dR_R0_ratio = current_amplitude / current_offset

    reference_condition = dict(material='Au')
    sin_data = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': current_offset + \
                current_amplitude * np.array([0, -1, 0, 1, 0]),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    test_data = {
        convert_name('TEST1~wavelength=700nm~material=Au'): sin_data,
        convert_name('TEST1~wavelength=750nm~material=Au'): sin_data,
        convert_name('TEST1~wavelength=700nm~material=Al'): sin_data,
        convert_name('TEST1~wavelength=750nm~material=Al'): sin_data,
    }
    exp = Experiment(
            name='TEST1', kind='test',
            wavelength=wavelength, material=material, gain=gain)
    exp.data = test_data

    R0_actual, dR_actual, inoise_actual = exp.process_photocurrent(
            reference_condition=reference_condition)
    R0_desired = {
        convert_name('TEST1~wavelength=700nm~material=Au'): 0.93329,
        convert_name('TEST1~wavelength=750nm~material=Au'): 0.948615,
        convert_name('TEST1~wavelength=700nm~material=Al'): 0.93329,
        convert_name('TEST1~wavelength=750nm~material=Al'): 0.948615,
    }
    dR_desired = {
        convert_name('TEST1~wavelength=700nm~material=Au'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=750nm~material=Au'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=700nm~material=Al'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=750nm~material=Al'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
    }
    inoise_desired = {
        convert_name('TEST1~wavelength=700nm~material=Au'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~wavelength=750nm~material=Au'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~wavelength=700nm~material=Al'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~wavelength=750nm~material=Al'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
    }
    assertDataDictEqual(R0_actual, R0_desired)
    assertDataDictEqual(dR_actual, dR_desired)
    assertDataDictEqual(inoise_actual, inoise_desired)
