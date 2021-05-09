import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
from sciparse import assert_equal_qt
from xsugar import sum_group, average_group, simplify_index, ureg, drop_redundants

@pytest.fixture
def data_1var():
    inner_data_1 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1, 2, 3]})
    inner_data_2 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [2, 3, 4]})
    nested_data = pd.DataFrame(
            index=pd.Index([1, 2], name='replicate'),
            data={'photocurrent': [inner_data_1, inner_data_2]})
    yield nested_data

@pytest.fixture
def data_1var_summed():
    summed_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [3, 5, 7]})
    yield summed_data

@pytest.fixture
def data_1var_averaged():
    summed_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1.5, 2.5, 3.5]})
    yield summed_data

@pytest.fixture
def data_2var():
    inner_data_1 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1, 2, 3]})
    inner_data_2 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [2, 3, 4]})
    nested_data = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[1, 2, 1, 2], [0, 0, 1, 1]],
            names=['wavelength', 'replicate']),
            data={'photocurrent':
                [inner_data_1, inner_data_2,
                inner_data_1, inner_data_2]})
    yield nested_data

@pytest.fixture
def data_2var_summed():
    summed_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [6, 10, 14]})
    yield summed_data

@pytest.fixture
def data_2var_averaged():
    summed_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1.5, 2.5, 3.5]})
    yield summed_data

# The index for this is a bit of a hack,
# I'm not even sure what it should be. 
def test_sum_scalar_group():
    data = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[1, 2], [0, 0]]),
            data={'current (nA)': [0, 1]})
    desired_sum = 1 * ureg.nA
    actual_sum = sum_group(data)

    assert_equal_qt(actual_sum, desired_sum)

def test_average_group_scalar():
    data = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[1, 2], [0, 0]]),
            data={'current (nA)': [0, 1]})
    desired_average = 0.5 * ureg.nA
    actual_average = average_group(data)

    assert_equal_qt(actual_average, desired_average)

def test_sum_group_1var_first(data_1var, data_1var_summed):
    actual_data = sum_group(data_1var, summing_type='first')
    desired_data = data_1var_summed
    assert_frame_equal(actual_data, desired_data)

def test_sum_group_1var_last(data_1var, data_1var_summed):
    actual_data = sum_group(data_1var, summing_type='last')
    desired_data = data_1var_summed
    assert_frame_equal(actual_data, desired_data)

def test_average_group_1var_first(data_1var, data_1var_averaged):
    desired_data = data_1var_averaged
    actual_data = average_group(data_1var, averaging_type='first')
    assert_frame_equal(actual_data, desired_data)

def test_sum_group_2var(data_2var, data_2var_summed):
    desired_data = data_2var_summed
    actual_data = sum_group(data_2var)
    assert_frame_equal(actual_data, desired_data)

def test_average_group_2var(data_2var, data_2var_averaged):
    desired_data = data_2var_averaged
    actual_data = average_group(data_2var)
    assert_frame_equal(actual_data, desired_data)

def test_simplify_index_1D():
    index = pd.MultiIndex.from_tuples(((1,), (2,)), names=['wavelength'])
    desired_index = pd.Index([1, 2], name='wavelength')
    actual_index = simplify_index(index)
    assert_index_equal(actual_index, desired_index)

def test_remove_duplicate_indices_multi():
    index_tuples = ((1, 25), (1, 35))
    duplicate_index = pd.MultiIndex.from_tuples(index_tuples, names=['wavelength', 'temperature'])

    desired_tuples = ((25,), (35,))
    desired_index = pd.Index([25, 35], name='temperature')
    actual_index = drop_redundants(duplicate_index)
    assert_index_equal(actual_index, desired_index)

def test_drop_redundants_single():
    index = pd.Index([1, 1, 2, 2])
    desired_index = pd.Index([1, 2])
    actual_index = drop_redundants(index)
    assert_index_equal(actual_index, desired_index)

def test_drop_redundants_multi_oneitem():
    index = pd.MultiIndex.from_tuples(((1,2),), names=['wavelength', 'temperature'])
    desired_index = index
    actual_index = drop_redundants(index)
    assert_index_equal(actual_index, desired_index)
