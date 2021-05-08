import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from xsugar import average_nested_group

def test_average_nested_group_1var():
    inner_data_1 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1, 2, 3]})
    inner_data_2 = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [2, 3, 4]})
    averaged_data = pd.DataFrame(
            data={'Time (ms)': [0, 1, 2],
            'Current (nA)': [1.5, 2.5, 3.5]})

    nested_data = pd.DataFrame(
            index=pd.Index([1, 2], name='replicate'),
            data={'photocurrent': [inner_data_1, inner_data_2]})
    grouped_data = nested_data.groupby('replicate')
    desired_data = pd.DataFrame(
            index=pd.Index([0]),
            data=averaged_data)
    actual_data = average_nested_group(grouped_data)

    assert_frame_equal(actual_data, desired_data)
