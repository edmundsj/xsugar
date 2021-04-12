from liapy import LIA
from sciparse import frequency_bin_size, column_from_unit, cname_from_unit, is_scalar
from spectralpy import power_spectrum
from xsugar import ureg
import numpy as np

def dc_photocurrent(data, cond):
    voltages = column_from_unit(data, ureg.mV)
    return (voltages.mean() / cond['gain']).to(ureg.nA)

def modulated_photocurrent(data, cond):
    """
    Returns the RMS value of the modulated photocurrent given the system gain and a dataset using lock-in amplification.
    """
    lia = LIA(data=data)
    if 'sync_phase_delay' in cond:
        sync_phase_delay = cond['sync_phase_delay']
    else:
        sync_phase_delay = np.pi
    extracted_voltage = lia.extract_signal_amplitude(
            mode='rms', sync_phase_delay=sync_phase_delay)
    extracted_current = (extracted_voltage / cond['gain']).to(ureg.pA)
    return extracted_current

def noise_current(data, cond):
    data_power = power_spectrum(data)
    column_name = cname_from_unit(data_power, ureg.Hz)
    if 'filter_cutoff' in cond:
        filter_cutoff = cond['filter_cutoff'].to(ureg.Hz).magnitude
    else:
        filter_cutoff = 200 # Default to 200Hz
    filtered_power = data_power[data_power[column_name] > filter_cutoff]
    average_noise_power= \
        column_from_unit(filtered_power, ureg.V ** 2).mean()
    bin_size = frequency_bin_size(filtered_power)
    noise_psd = average_noise_power / bin_size / (cond['gain'])**2
    noise_psd = noise_psd.to(ureg.A ** 2 / ureg.Hz)
    return noise_psd

def inoise_func_dBAHz(xdata, R=1*ureg.Mohm):
    """
    Returns the current noise density in dBA/Hz of
    a 1Mohm resistor

    :param R: Resistance of resistor
    """
    T = 300 * ureg.K
    inoise = (4 * ureg.k * T / R).to(ureg.A**2/ureg.Hz)
    inoise = 10*np.log10(inoise.m) # Current noise PSD in dBA/Hz
    if is_scalar(xdata):
        return inoise
    elif isinstance(xdata, (list, np.ndarray, tuple)):
        return np.ones(len(xdata))*inoise
