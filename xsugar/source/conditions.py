from xsugar import ureg
import copy
import pint

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
