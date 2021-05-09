import numpy as np
import pandas as pd
from sciparse import is_scalar, title_to_quantity
from xsugar import is_collection

def sum_group_scalar(grouped_data):
    """
    Sums a group of scalar data organized into a pandas array, summing by column.
    :param grouped_data: Pandas dataframe of any number of columns to sum.
    """
    iterable_data = grouped_data.groupby(grouped_data.index)
    summed_data = grouped_data.sum()

    data_units = [title_to_quantity(x) for x in summed_data.index]
    return_data = []
    for data, unit in zip(summed_data.values, data_units):
        if unit.dimensionless == True:
            return_data.append(data)
        else:
            return_data.append(data*unit)
    if len(return_data) == 1:
        return_data = return_data[0]
    return return_data

def sum_group(grouped_data, summing_type='first'):
    iterable_data = grouped_data.groupby(grouped_data.index)
    first_data = iterable_data.first().iloc[0, 0]
    if isinstance(first_data, pd.DataFrame):
        return sum_group_pandas(grouped_data, summing_type=summing_type)
    elif is_collection(first_data) or is_scalar(first_data):
        return sum_group_scalar(grouped_data)

def sum_group_pandas(grouped_data, summing_type='first'):
    """
    Sums all the dataframes given in the grouped data.

    :param grouped_data: pandas DataFrame of dataframes to average
    :param summing_type: Whether to sum all but the first column, or only the last column
    """
    iterable_data = grouped_data.groupby(grouped_data.index)
    data_to_sum = None
    first_data = iterable_data.first().iloc[0, 0]

    for ind, data in iterable_data:
        if len(data) > 1:
            raise ValueError('There appears to be more than one data frame in data. Cannot properly average.')
        data = data.iloc[0, 0] # extract the underlying data
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

def average_group(grouped_data, averaging_type='first'):
    iterable_data = grouped_data.groupby(grouped_data.index)
    first_data = iterable_data.first().iloc[0, 0]
    if isinstance(first_data, pd.DataFrame):
        return average_group_pandas(grouped_data,
                averaging_type=averaging_type)
    elif is_scalar(first_data):
        return average_group_scalar(grouped_data)

def average_group_scalar(grouped_data):
    summed_group = sum_group_scalar(grouped_data)
    if is_collection(summed_group):
        for i, val in enumerate(summed_group):
            summed_group[i] = val / len(grouped_data)
    else:
        summed_group /= len(grouped_data)
    return summed_group

def average_group_pandas(grouped_data, averaging_type='first'):
    summed_data = sum_group(grouped_data, summing_type=averaging_type)
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

def drop_redundants(x):
    """
    Drops redundant values from a collection. If a pandas MultiIndex, drops values from the index which are the same in every index tuple.
    """
    if len(x) == 1:
        return x
    if isinstance(x, (list, np.ndarray)):
        return_vals = []
        for item in x:
            if item not in return_vals:
                return_vals.append(item)
    elif isinstance(x, pd.Index):
        first_val = x[0]
        if is_collection(first_val):
            redundant_indices = np.ones(len(first_val), dtype=bool)
            for tup in x:
                for i, (v1, v2) in enumerate(zip(tup, first_val)):
                    if v1 != v2:
                        redundant_indices[i] = False
            new_index = x
            reversed_i = [i for i in range(len(redundant_indices))]
            reversed_i.reverse()
            for i, val in zip(reversed_i, np.flip(redundant_indices)):
                if val == True:
                    new_index = new_index.droplevel(i)
            return_vals = new_index
        else:
             return_vals = x.drop_duplicates()

    return return_vals
