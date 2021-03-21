from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
from itertools import zip_longest
from pandas import DataFrame

def assertDataDictEqual(data_dict_actual, data_dict_desired):
    assert_equal(type(data_dict_actual), dict)
    assert_equal(type(data_dict_desired), dict)
    for actual_name, desired_name in \
        zip_longest(data_dict_actual.keys(),
                    data_dict_desired.keys()):
        assert_equal(actual_name, desired_name)

    for actual_data, desired_data in \
         zip_longest(data_dict_actual.values(),
                     data_dict_desired.values()):
        if isinstance(actual_data, DataFrame):
            assert_frame_equal(actual_data, desired_data)
        elif isinstance(actual_data, dict):
            assertDataDictEqual(actual_data, desired_data)
        else:
            assert_equal(actual_data, desired_data)
