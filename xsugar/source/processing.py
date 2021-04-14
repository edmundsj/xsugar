from liapy import LIA
from sciparse import frequency_bin_size, column_from_unit, cname_from_unit, is_scalar
from spectralpy import power_spectrum
from xsugar import ureg, condition_is_subset, condition_from_name
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
            mode='amplitude', sync_phase_delay=sync_phase_delay)
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

def match_theory_data(curve_name, sim_exp):
    """
    :param curve_dict: Data dictionary to be plotted as a single curve. Assumes dictionary has a single key-value pair of the form name: data
    :param sim_exp: Experiment from which to draw the theoretical data

    """
    curve_condition = condition_from_name(curve_name)

    sim_exp_partial_conditions = [condition_from_name(n) for n in sim_exp.data.keys()]
    is_subset = [condition_is_subset(c, curve_condition) \
        for c in sim_exp_partial_conditions]
    subset_indices = [i for i, val in enumerate(is_subset) if val]
    matching_conditions = [sim_exp.conditions[i] for i in subset_indices]
    matching_names = [sim_exp.nameFromCondition(c) for c in matching_conditions]

    data_values = [sim_exp.data[name] for name in matching_names]

    if len(subset_indices) > 1:
        warnings.warn('Warning: more than one theoretical dataset matches the desired dataset. Matches are ')

    if len(data_values) >= 1:
        return data_values[0]
    elif len(data_values) == 0:
        return None
