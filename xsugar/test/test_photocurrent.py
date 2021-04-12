import pytest
from xsugar import ureg, dc_photocurrent, modulated_photocurrent, noise_current, Experiment
from sciparse import assert_equal_qt, assert_allclose_qt, assertDataDictEqual
from sugarplot import assert_figures_equal, plt, Figure
import pandas as pd
import numpy as np
import os

@pytest.fixture
def sim_exp():
    file_location = os.path.dirname(os.path.abspath(__file__))
    base_path = file_location + '/data'
    sim_exp = Experiment(name='REFL2', kind='test', base_path=base_path)
    return sim_exp

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
            atol=1e-30, rtol=2e-2)

def test_process_photocurrent_simple(convert_name, sim_exp):
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
        convert_name('TEST1~material=Au~wavelength=700nm'): sin_data,
        convert_name('TEST1~material=Au~wavelength=750nm'): sin_data,
        convert_name('TEST1~material=Al~wavelength=700nm'): sin_data,
        convert_name('TEST1~material=Al~wavelength=750nm'): sin_data,
    }
    exp = Experiment(
            name='TEST1', kind='test',
            wavelength=wavelength, material=material, gain=gain)
    exp.data = test_data

    R0_actual, dR_actual, inoise_actual = exp.process_photocurrent(
            reference_condition=reference_condition, sim_exp=sim_exp)
    R0_desired = {
        convert_name('TEST1~material=Au~wavelength=700nm'): 0.93329,
        convert_name('TEST1~material=Au~wavelength=750nm'): 0.948615,
        convert_name('TEST1~material=Al~wavelength=700nm'): 0.93329,
        convert_name('TEST1~material=Al~wavelength=750nm'): 0.948615,
    }
    dR_desired = {
        convert_name('TEST1~material=Au~wavelength=700nm'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~material=Au~wavelength=750nm'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~material=Al~wavelength=700nm'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~material=Al~wavelength=750nm'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
    }
    inoise_desired = {
        convert_name('TEST1~material=Au~wavelength=700nm'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~material=Au~wavelength=750nm'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~material=Al~wavelength=700nm'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
        convert_name('TEST1~material=Al~wavelength=750nm'): \
            8.000000000000231e-22 * ureg.A ** 2 / ureg.Hz,
    }
    assertDataDictEqual(R0_actual, R0_desired)
    assertDataDictEqual(dR_actual, dR_desired)
    assertDataDictEqual(inoise_actual, inoise_desired)

def test_plot_photocurrent_simple(convert_name, sim_exp):
    """
    Verifies that, given a sinusoidal input with a known offset and amplitude, the correct data is generated.
    """
    wavelength = np.array([700, 750]) * ureg.nm
    material = ['Au', 'Al']
    gain = 1 * ureg.Mohm
    current_offset = 100*ureg.nA # nA
    current_amplitude = 10*ureg.pA # 
    dR_R0_ratio = current_amplitude / current_offset

    reference_condition = dict(material='Au')
    sin_data = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': current_offset.to(ureg.nA).m + \
                current_amplitude.to(ureg.nA).m * \
                np.array([0, -1, 0, 1, 0]),
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
    (R0_figs_actual, _, dR_figs_actual, _, inoise_figs_actual, _) =  \
         exp.plot_photocurrent(
                 reference_condition=reference_condition, sim_exp=sim_exp)

    Au_R0_data = sim_exp.data['REFL2~material=Au~modulation_voltage=0V~spectra=R0']
    Al_R0_data = sim_exp.data['REFL2~material=Al~modulation_voltage=0V~spectra=R0']
    R0_fig_desired = Figure()
    R0_ax = R0_fig_desired.subplots()
    R0_ax.plot([700, 750], [0.93329, 0.948615])
    R0_ax.plot(Au_R0_data['Wavelength (nm)'].values, Au_R0_data['R'].values, linestyle='dashed')
    R0_ax.plot([700, 750], [0.93329, 0.948615])
    R0_ax.plot(Al_R0_data['Wavelength (nm)'].values, Al_R0_data['R'].values, linestyle='dashed')
    R0_ax.set_xlabel('wavelength (nm)')
    R0_ax.set_ylabel(r'$R_0$')
    R0_ax.set_xlim(700*0.9, 750*1.1)

    # NOTE - This is the pk-pk data for 10V amplitude. To convert to rms,
    # We need only to divide by 2 * sqrt(2)

    dR_fig_desired = Figure()
    dR_ax = dR_fig_desired.subplots()
    dR_ax.plot([700, 750],
            [0.93329 / np.sqrt(2) * dR_R0_ratio,
            0.948615 / np.sqrt(2) * dR_R0_ratio])
    dR_ax.plot([700, 750],
            [0.93329 / np.sqrt(2) * dR_R0_ratio,
            0.948615 / np.sqrt(2) * dR_R0_ratio])
    dR_ax.set_xlabel('wavelength (nm)')
    dR_ax.set_ylabel(r'$\Delta R_{rms}$')

    inoise_fig_desired = Figure()
    inoise_ax_desired = inoise_fig_desired.subplots()
    inoise_ax_desired.plot([700, 750],
            [-250.9691, -250.9691])
    inoise_ax_desired.plot([700, 750],
            [-257.8073547127414, -257.8073547127414], linestyle='dashed')
    inoise_ax_desired.plot([700, 750],
            [-250.9691, -250.9691])
    inoise_ax_desired.plot([700, 750],
            [-257.8073547127414, -257.8073547127414], linestyle='dashed')

    inoise_ax_desired.set_xlabel('wavelength (nm)')
    inoise_ax_desired.set_ylabel('Power (dBA/Hz)')

    assert_figures_equal(inoise_figs_actual[0], inoise_fig_desired, atol=1e-6)
    assert_figures_equal(R0_figs_actual[0], R0_fig_desired, atol=1e-6)
    assert_figures_equal(dR_figs_actual[0], dR_fig_desired, atol=1e-6)

def test_plot_photocurrent_realistic(convert_name, sim_exp):
    """
    Verifies that, given a sinusoidal input with a known offset and amplitude, the correct data is generated.
    """
    wavelength = np.array([850, 1150]) * ureg.nm
    modulation_voltage = ureg.V * np.array([10, 20])
    gain = 10 * ureg.Mohm
    current_offset = 100*ureg.nA # nA
    current_amplitude = 10*ureg.pA # 
    dR_R0_ratio = current_amplitude / current_offset

    reference_condition = dict(material='Au')
    Au_data = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': (gain*current_offset).to(ureg.mV).m*np.ones(5),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    sin_data_10V_850nm = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': \
                    0.0261059412418133/0.96400635435947* \
                    (gain*current_offset).to(ureg.mV).m + \
                (gain*current_amplitude).to(ureg.mV).m * \
                np.array([0, -1, 0, 1, 0]),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    sin_data_10V_1150nm = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': \
                    0.283227973929987/0.975276963043909* \
                    (gain*current_offset).to(ureg.mV).m + \
                    (gain*current_amplitude).to(ureg.mV).m * \
                np.array([0, -1, 0, 1, 0]),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    sin_data_20V_850nm = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)':
                    0.0261059412418133/0.96400635435947*  \
                    (gain*current_offset).to(ureg.mV).m + \
                2 * (gain*current_amplitude).to(ureg.mV).m * \
                np.array([0, -1, 0, 1, 0]),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    sin_data_20V_1150nm = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)':
                    0.283227973929987/0.975276963043909* \
                    (gain*current_offset).to(ureg.mV).m + \
                2 * (gain*current_amplitude).to(ureg.mV).m * \
                np.array([0, -1, 0, 1, 0]),
            'Sync': np.array([1, 0, 0, 0, 1]),
            })
    test_data = {
        convert_name('TEST1~wavelength=850nm~material=Au~modulation_voltage=0V'): Au_data,
        convert_name('TEST1~wavelength=1150nm~material=Au~modulation_voltage=0V'): Au_data,
        convert_name('TEST1~wavelength=850nm~material=AlN~modulation_voltage=10V'): sin_data_10V_850nm,
        convert_name('TEST1~wavelength=1150nm~material=AlN~modulation_voltage=10V'): sin_data_10V_1150nm,
        convert_name('TEST1~wavelength=850nm~material=AlN~modulation_voltage=20V'): sin_data_20V_850nm,
        convert_name('TEST1~wavelength=1150nm~material=AlN~modulation_voltage=20V'): sin_data_20V_1150nm,
    }
    exp = Experiment(
            name='TEST1', kind='test',
            wavelength=wavelength,
            modulation_voltage=modulation_voltage,
            material=['AlN'],
            gain=gain)
    exp.append_condition(
        comb_type='cartesian',
        wavelength=wavelength,
        material='Au',
        modulation_voltage=0*ureg.V,
        gain=gain)
    exp.data = test_data

    R0_desired = {
        convert_name('TEST1~wavelength=850nm~modulation_voltage=0V'): 0.0261059412418133,
        convert_name('TEST1~wavelength=1150nm~modulation_voltage=0V'): 0.283227973929987,
        convert_name('TEST1~wavelength=850nm~modulation_voltage=0V'): 0.0261059412418133,
        convert_name('TEST1~wavelength=1150nm~modulation_voltage=0V'): 0.283227973929987,
    }

    dR_desired = {
        convert_name('TEST1~wavelength=850nm~modulation_voltage=10V'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=1150nm~modulation_voltage=10V'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=850nm~modulation_voltage=20V'): \
            0.93329 / np.sqrt(2) * dR_R0_ratio,
        convert_name('TEST1~wavelength=1150nm~modulation_voltage=20V'): \
            0.948615 / np.sqrt(2) * dR_R0_ratio,
    }
    (R0_figs_actual, _, dR_figs_actual, _, inoise_figs_actual, _) =  \
         exp.plot_photocurrent(
                 reference_condition=reference_condition, sim_exp=sim_exp,
                 c_axis_include='modulation_voltage')

    AlN_R0_data = sim_exp.data['REFL2~material=AlN~modulation_voltage=0V~spectra=R0']
    R0_fig_desired = Figure()
    R0_ax = R0_fig_desired.subplots()
    R0_ax.plot([850, 1150], [0.026106, 0.283228])
    R0_ax.plot(AlN_R0_data['Wavelength (nm)'].values, AlN_R0_data['R'].values, linestyle='dashed')
    R0_ax.plot([850, 1150], [0.026106, 0.283228])
    R0_ax.plot(AlN_R0_data['Wavelength (nm)'].values, AlN_R0_data['R'].values, linestyle='dashed')
    R0_ax.set_xlabel('wavelength (nm)')
    R0_ax.set_ylabel(r'$R_0$')
    R0_ax.set_xlim(850*0.9, 1150*1.1)

    AlN_dR_data = sim_exp.data['REFL2~material=AlN~modulation_voltage=10V~spectra=deltaR']
    # NOTE - This is the pk-pk data for 10V amplitude. To convert to rms,
    # We need only to divide by 2 * sqrt(2)
    AlN_dR_data['R'] /= (2 * np.sqrt(2))

    dR_fig_desired = Figure()
    dR_ax = dR_fig_desired.subplots()
    dR_ax.plot([850, 1150],
            [0.96400635435947 / np.sqrt(2) * dR_R0_ratio,
            0.975276963043909 / np.sqrt(2) * dR_R0_ratio])
    dR_ax.plot(AlN_dR_data['Wavelength (nm)'].values, AlN_dR_data['R'].values,
            linestyle='dashed')
    dR_ax.plot([850, 1150],
            [2*0.96400635435947 / np.sqrt(2) * dR_R0_ratio,
            2*0.975276963043909 / np.sqrt(2) * dR_R0_ratio])
    dR_ax.plot(AlN_dR_data['Wavelength (nm)'].values, 2*AlN_dR_data['R'].values,
            linestyle='dashed')
    dR_ax.set_xlabel('wavelength (nm)')
    dR_ax.set_ylabel(r'$\Delta R_{rms}$')
    dR_ax.set_xlim(850*0.9, 1150*1.1)

    inoise_fig_desired = Figure()
    inoise_ax_desired = inoise_fig_desired.subplots()
    inoise_ax_desired.plot([850, 1150],
            [-250.9691, -250.9691])
    inoise_ax_desired.plot([850, 1150],
            [-257.8073547127414, -257.8073547127414], linestyle='dashed')
    inoise_ax_desired.plot([850, 1150],
            [-244.9485, -244.9485])
    inoise_ax_desired.plot([850, 1150],
            [-257.8073547127414, -257.8073547127414], linestyle='dashed')

    inoise_ax_desired.set_xlabel('wavelength (nm)')
    inoise_ax_desired.set_ylabel('Power (dBA/Hz)')

    assert_figures_equal(inoise_figs_actual[1], inoise_fig_desired, atol=1e-6)
    assert_figures_equal(R0_figs_actual[1], R0_fig_desired, atol=1e-6)
    assert_figures_equal(dR_figs_actual[1], dR_fig_desired, atol=1e-6)
