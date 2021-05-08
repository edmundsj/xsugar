import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
from xsugar import average_nested_group, sum_nested_group, simplify_index

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

def test_sum_nested_group_1var_first(data_1var, data_1var_summed):
    actual_data = sum_nested_group(data_1var, summing_type='first')
    desired_data = data_1var_summed
    assert_frame_equal(actual_data, desired_data)

def test_sum_nested_group_1var_last(data_1var, data_1var_summed):
    actual_data = sum_nested_group(data_1var, summing_type='last')
    desired_data = data_1var_summed
    assert_frame_equal(actual_data, desired_data)

def test_average_nested_group_1var_first(data_1var, data_1var_averaged):
    desired_data = data_1var_averaged
    actual_data = average_nested_group(data_1var, averaging_type='first')
    assert_frame_equal(actual_data, desired_data)

def test_sum_nested_group_2var(data_2var, data_2var_summed):
    desired_data = data_2var_summed
    actual_data = sum_nested_group(data_2var)
    assert_frame_equal(actual_data, desired_data)

def test_average_nested_group_2var(data_2var, data_2var_averaged):
    desired_data = data_2var_averaged
    actual_data = average_nested_group(data_2var)
    assert_frame_equal(actual_data, desired_data)

def test_simplify_index():
    index = pd.MultiIndex.from_tuples(((1,), (2,)), names=['wavelength'])
    desired_index = pd.Index([1, 2], name='wavelength')
    actual_index = simplify_index(index)
    assert_index_equal(actual_index, desired_index)
