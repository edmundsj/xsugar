from liapy import LIA
from sciparse import frequency_bin_size, column_from_unit, cname_from_unit
from spectralpy import power_spectrum
from xsugar import ureg

def dc_photocurrent(data, cond):
    voltages = column_from_unit(data, ureg.mV)
    return (voltages.mean() / cond['gain']).to(ureg.nA)

def modulated_photocurrent(data, cond):
    """
    Returns the RMS value of the modulated photocurrent given the system gain and a dataset using lock-in amplification.
    """
    lia = LIA(data=data)
    extracted_voltage = lia.extract_signal_amplitude(mode='rms')
    extracted_current = (extracted_voltage / cond['gain']).to(ureg.pA)
    return extracted_current

def noise_current(data, cond):
    data_power = power_spectrum(data)
    column_name = cname_from_unit(data_power, ureg.Hz)
    filter_cutoff = cond['filter_cutoff'].to(ureg.Hz).magnitude
    filtered_power = data_power[data_power[column_name] > filter_cutoff]
    average_noise_power= \
        column_from_unit(filtered_power, ureg.V ** 2).mean()
    bin_size = frequency_bin_size(filtered_power)
    noise_psd = average_noise_power / bin_size / (cond['gain'])**2
    noise_psd = noise_psd.to(ureg.A ** 2 / ureg.Hz)
    return noise_psd
