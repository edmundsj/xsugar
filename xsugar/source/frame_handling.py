import numpy as np
import pandas as pd

def sum_nested_group(grouped_data, summing_type='first'):
    """
    Sums all the dataframes given in the grouped data.

    :param grouped_data: pandas DataFrame of dataframes to average
    :param summing_type: Whether to sum all but the first column, or only the last column
    """
    data_to_sum = None

    iterable_data = grouped_data.groupby(grouped_data.index)
    for ind, data in iterable_data:
        if len(data) > 1:
            raise ValueError('There appears to be more than one data frame in data. Cannot properly average.')
        data = data.iloc[0, 0] # extract the actual frame
        if summing_type == 'first':
            new_data_to_sum = data.iloc[:, 1:]
        elif summing_type == 'last':
            new_data_to_sum = data.iloc[:, [-1]]
        elif summing_type == 'all':
            new_data_to_sum = data
        else:
            raise ValueError(f'Invalid summiing type {summing_type}. Available types are "first", "last", and "all"')
        if data_to_sum is None:
            data_to_sum = new_data_to_sum
        else:
            data_to_sum += new_data_to_sum

    if summing_type == 'first':
        columns_to_insert = [data.columns[0]]
        values_to_insert = [data.iloc[:,0].values]
    elif summing_type == 'last':
        columns_to_insert = data.columns[:-1].values
        values_to_insert = [data.iloc[:,:-1].values]
        columns_to_insert = np.flip(columns_to_insert)
        values_to_insert = np.flip(values_to_insert, axis=0)
    else:
        columns_to_insert = []
        values_to_insert = []

    for c, val in zip(columns_to_insert, values_to_insert):
        data_to_sum.insert(loc=0, column=c, value=val)

    return data_to_sum

    # Re-insert the column we stripped out

def average_nested_group(grouped_data, averaging_type='first'):
    summed_data = sum_nested_group(
            grouped_data, summing_type=averaging_type)
    averaged_data = summed_data
    if averaging_type == 'first':
        averaged_data.iloc[:,1:] = summed_data.iloc[:,1:] / len(grouped_data)
    elif averaging_type == 'last':
        averaged_data.iloc[:,-1] = summed_data.iloc[:,-1] / len(grouped_data)
    elif averaging_type == 'all':
        averaged_data = summed_data / len(grouped_data)
    else:
        raise ValueError(f'Invalid averaging type {averaging_type}. Available types are "first", "last", and "all"')
    return averaged_data

def simplify_index(index):
    return_index = index
    if isinstance(index, pd.MultiIndex):
        first_item = index[0]
        if len(first_item) == 1:
            return_index = pd.Index([elem[0] for elem in index],
                    name=index.names[0])
    return return_index
