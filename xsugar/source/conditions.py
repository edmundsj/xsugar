from xsugar import ureg
import copy
import pint
import numpy as np
import pandas as pd

def is_collection(x):
    is_iterable = hasattr(x, '__iter__')
    if isinstance(x, str):
        is_iterable = False
    return is_iterable

def drop_redundants(x):
    """
    Drops redundant values from a collection. If a pandas MultiIndex, drops values from the index which are the same in every index tuple.
    """
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

def factors_from_condition(cond):
    factors = [x for x in cond.keys()]
    return factors

def get_partial_condition(cond,
        exclude_factors=[], include_factors=[]):
    """
    Gets a subset of a full condition
    """
    new_cond = copy.copy(cond)
    if isinstance(exclude_factors, str):
        exclude_factors = [exclude_factors]
    if isinstance(include_factors, str):
        include_factors = [include_factors]
    if include_factors:
        new_cond = {}
        for factor in include_factors:
            if factor in cond.keys():
                new_cond[factor] = cond[factor]

    if exclude_factors and not include_factors:
        new_cond = copy.copy(cond)
        for factor in exclude_factors:
            if factor in new_cond.keys():
                del new_cond[factor]

    return new_cond

def condition_is_subset(subset_cond, superset_cond):
    """
    :param subset_cond: Hypothetical subset condition
    :param superset_cond: Hypothetical superset condition
    :returns: True or False, whether the subset condition is a subset of the superset condition
    """

    subset_items = subset_cond.items()
    superset_items = superset_cond.items()
    is_subset = all(item in superset_items for item in subset_items)
    return is_subset

def condition_from_name(
        name, major_separator='~', minor_separator='=',
        metadata={}, constants={},
        full_condition=True):

    sub_components = name.split(major_separator)
    name_id = sub_components[0].split(minor_separator)
    sub_components = sub_components[1:]
    cond = {}

    cond.update({sub.split(minor_separator)[0]:sub.split(minor_separator)[1] for sub in sub_components})

    for key, val in cond.items():
        try:
            if val == 'c' or val == 'dR':
                raise pint.errors.UndefinedUnitError
            cond[key] = ureg.parse_expression(val)
        except pint.errors.UndefinedUnitError:
            cond[key] = val
    if full_condition:
        cond.update(constants)
        if name in metadata.keys():
            cond.update(metadata[name])
    return cond

def conditions_from_index(index):
    conds = []
    cond_names = index.names
    cond_val_pairs = index.values
    for cond_val_pair in cond_val_pairs:
        if len(cond_names) > 1:
            cond = {k: v for k, v in zip(cond_names, cond_val_pair)}
        else:
            cond = {cond_names[0]: cond_val_pair}
        conds.append(cond)
    if len(conds) == 0:
        raise ValueError(f'No conditions could be found from index {index}')
    return conds
