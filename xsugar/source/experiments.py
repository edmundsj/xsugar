import itertools
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pint
from matplotlib.figure import Figure
from sugarplot import plt, prettifyPlot, default_plotter, power_spectrum_plot, normalize_reflectance
from spectralpy import power_spectrum
from sciparse import parse_xrd, parse_default, is_scalar, dict_to_string, title_to_quantity, to_standard_quantity, quantity_to_title
from itertools import permutations
from xsugar import ureg, dc_photocurrent, modulated_photocurrent, noise_current, inoise_func_dBAHz, factors_from_condition, get_partial_condition, condition_is_subset, condition_from_name, match_theory_data
from xsugar import conditions_from_index, drop_redundants, is_collection, average_group, sum_group, simplify_index
import copy

class Experiment:
    """
    :param kwargs: Any number of keyword arguments to pass into the experiment. These can be scalars/strings, or lists. If lists, all cartesian products of all possible experiments will be run.
    :param name: The full name of the specific experiment (i.e. LIA1-1)
    :param kind: A string of the kind of experiment we are carrying out (i.e. XRD, simulation, etc.
    :param measure_func: The function to execute which returns data given a particular experimental condition. Must take the experimental condition as an argument
    :param base_path: The absolute or relative base path of all structures.
    :param verbose: Verbose output enable/disable
    """

    def __init__(self, name, kind, measure_func=None,
                 ident='', verbose=False,
                 base_path=None, **kwargs):
        if not base_path:
            base_path = str(Path.home())
            if 'LOGNAME' in os.environ:
                if os.environ['LOGNAME'] == 'jordan.e':
                    base_path += '/Google Drive/GTD/PhD/docs/pockels_modulator'
        self.major_separator = '~'
        self.minor_separator = '='
        self.name = name
        self.ident = ident
        self.base_path = base_path
        self.data = {}
        self.metadata = {}
        self.verbose = verbose
        self.measure_func = measure_func
        if measure_func:
            self.measure_name = measure_func.__name__
        else:
            self.measure_name = None

        if kind == 'designs' or kind == 'design':
            self.data_full_path = base_path + '/designs/'+ name + '/data/'
            self.figures_full_path = base_path + '/designs/' + name + '/figures/'
        else:
            self.data_full_path = self.base_path + '/data/' + kind + '/' + name + '/'
            self.figures_full_path = self.base_path + '/figures/' + kind + '/' + name + '/'
            self.process_full_path = self.base_path + '/processing_code/' + kind + '/'
        self.factors = {k:v for k, v in kwargs.items() \
             if isinstance(v, (list, np.ndarray))}
        for k, v in kwargs.items():
            if isinstance(v, pint.Quantity):
                if isinstance(v.magnitude, (np.ndarray, list)):
                    self.factors[k] = v
        self.constants = {k:v for k, v in kwargs.items() \
                 if k not in self.factors.keys()}

        self.conditions = self.generate_conditions(**self.factors)

        if not os.path.exists(self.data_full_path):
            os.makedirs(self.data_full_path)
        if not os.path.exists(self.figures_full_path):
            os.makedirs(self.figures_full_path)

    def Execute(self, **kwargs):
        if self.verbose == True: print(f"Executing experimnent {self.name}")
        for cond in self.conditions:
            self.executeExperimentCondition(cond, **kwargs)
            if self.verbose == True: print(f"Executing condition {cond}")

    def executeExperimentCondition(self, cond, **kwargs):
        """
        Executes the experiment with a given measure_function

        :param measure_function: function that executes the experiment. Must return a set of raw data (typically a pandas DataFrame)
        """
        condition_name = self.conditionToName(cond)
        data = self.measure_func(cond, **kwargs)
        self.data[condition_name] = data
        self.saveRawResults(data, cond)

# This is a mess. it should be refactored making use of external parsers in the sciparse library. Move all the unit stuff, and all the writing, out there.
    def saveRawResults(self, raw_data, cond):
        """
        Saves raw results from the experiment performed with a given condition. If the data is a scalar, saves the results in a single pandas array. If the data is itself a numpy/pandas array, saves the data in its own file for later analysis.

        :param cond: Experimental condition as a dictionary
        :param raw_data: Raw data as a pandas array to save
        """
        partial_filename = self.nameFromCondition(cond)
        self.data[partial_filename] = raw_data
        data_is_scalar = is_scalar(raw_data)
        data_is_pandas = isinstance(raw_data, pd.DataFrame)
        if data_is_pandas:
            full_filename = self.data_full_path + partial_filename + '.csv'
            parse_default(
                    full_filename,
                    data=raw_data, metadata=self.constants,
                    read_write='w')

        elif data_is_scalar:
            full_filename = self.data_full_path + self.name + '.csv'
            cond_partial = self.conditionFromName(partial_filename,
                    full_condition=False)
            if len(self.data) == 1:
                with open(full_filename, 'w') as fh:
                    metadata_line = dict_to_string(self.constants) + '\n'
                    fh.write(metadata_line)
                    for k, v in cond_partial.items():
                        if isinstance(v, pint.Quantity):
                            title_name = quantity_to_title(
                                    v, name=k)
                        else:
                            title_name = k
                        fh.write(title_name + ',')
                    if isinstance(raw_data, pint.Quantity):
                        data_name = quantity_to_title(
                                raw_data, name=self.measure_name)
                    else:
                        if self.measure_name:
                            data_name = self.measure_name
                        else:
                            data_name = 'Value'

                    fh.write(data_name + '\n')

            with open(full_filename, 'a') as fh:
                for v in cond_partial.values():
                    if isinstance(v, pint.Quantity):
                        quantity_name = str(v.magnitude)
                    else:
                        quantity_name = str(v)
                    fh.write(quantity_name + ',')
                if isinstance(raw_data, pint.Quantity):
                    raw_string = str(raw_data.magnitude)
                else:
                    raw_string = str(raw_data)

                fh.write(raw_string + '\n')
        else:
            raise ValueError(f'Cannot save data type {type(raw_data)}. Can only currently handle types of float, int, and pd.DataFrame')
        print(f'Results saved to {full_filename}')

    # This will be replaced with simple multi-indexing. No need for any craziness.
    def generate_conditions(self, comb_type='cartesian', **factors):
        """
        Generates a list of desired conditions from the specified factors and their levels.

        :param factors: dictionary with the factors and their desired levels
        :param comb_type: "cartesian" (outer product) - generates a full factorial experiment from the set of factors, "individual" creates an individual condition from the specified factor levels, "1to1" generates a 1-to-1 mapping of factor levels.
        """
        if not factors:
            factors = self.factors
        cond_list = []
        keys = list(factors.keys())
        vals = list(factors.values())
        for i, val in enumerate(vals):
            if is_scalar(val) or isinstance(val, str):
                vals[i] = [val]
        if comb_type == 'cartesian': # replace with pd.MultiIndex.from_product
            for cond in itertools.product(*vals):
                iterable_cond = dict(zip(keys, cond))
                full_cond = dict(iterable_cond, **self.constants)
                cond_list.append(full_cond)
        elif comb_type == 'individual':
            cond = factors
            full_cond = dict(cond, **self.constants)
            cond_list.append(full_cond)
        elif comb_type == '1to1' or comb_type == 'onetoone':
            first_factor_key = list(keys)[0]
            first_factor_val = list(vals)[0]
            cond_dict = {}
            for i in range(len(first_factor_val)):
                cond_dict[i] = {}
                for k, v in factors.items():
                    cond_dict[i][k] = v[i]
                cond_dict[i] = dict(cond_dict[i], **self.constants)
            cond_list = list(cond_dict.values())

        return cond_list

    def append_condition(self, comb_type='individual', **kwargs):
        """
        Appends a single condition to the existing list of conditions using the metadata already defined in the experiment initialization. Modifies in-place.

        :param kwargs: Keyword arguments with the values for the condition you want to append
        """
        extra_conds = self.generate_conditions(
                comb_type=comb_type, **kwargs)
        self.conditions.extend(extra_conds)

    def insert_condition(
            self, insertion_location, comb_type='individual', **kwargs):
        """
        Inserts a condition (or multiple conditions) at a specified index location in the list. Modifies in-place.

        :param insertion_location: Location of insertion in the conditions list (0 if at the beginning)
        :param kwargs: Keyword arguments of factors and their levels
        """
        extra_conds = self.generate_conditions(
                comb_type=comb_type, **kwargs)
        if insertion_location >= len(self.conditions):
            insertion_location = len(self.conditions) - 1
        self.conditions[insertion_location:insertion_location] = \
             extra_conds

    def get_conditions(self, data=None, exclude=[], full_condition=True):
        """
        Get all the conditions from a given set of data

        :param data_dict: The data you wish to extract conditions from
        :param exclude: The list of factors you would like to exclude from consideration.
        :param include: The complete list of factors you want included
        :returns conds: A list of conditions that match the inclusion and exclusion criteria
        """
        if data is None:
            data = self.data
        if isinstance(exclude, str):
            exclude = [exclude]
        conds = conditions_from_index(data.index)

        for cond in conds:
            for factor in exclude:
                cond.pop(factor, None) # Remove the condition
        conds = drop_redundants(conds)

        if full_condition == True:
            for cond in conds:
                cond.update(self.metadata)

        return conds

    def drop_name(self, name):
        return name.replace(self.name + self.major_separator, '')

    def prettify_name(self, name):
        cond = self.conditionFromName(name)
        new_cond = {}
        for k, v in cond.items():
            if v != 'x' and v != 'c':
                new_cond[k] = v

        name = self.nameFromCondition(new_cond)
        name = self.drop_name(name)
        name = name.replace('~', ', ')
        name = name.replace('_', ' ')
        return name

    def nameFromCondition(self, cond):
        """
        Generates filename for an experiment from a condition, returning a name
        with the metadata truncated, experiment name included

        :param condition: Condition to generate name for
        """
        partial_filename = self.name
        key_list = list(cond.keys())
        key_list.sort()
        for key in key_list:
            val = cond[key]
            if key not in self.constants.keys() and key not in \
            self.metadata.keys():
                partial_filename += self.major_separator + key + \
                    self.minor_separator
                if isinstance(val, pint.Quantity):
                    qt_str = '{:~}'.format(val).replace(' ', '')
                    partial_filename += qt_str
                else:
                    partial_filename += str(val)

        return partial_filename

    def conditionToName(self, cond):
        return self.nameFromCondition(cond)

    def nameToCondition(self, cond, full_condition=True):
        return self.conditionFromName(cond, full_condition=full_condition)

    def validateName(self, name):
        sub_components = name.split(self.major_separator)
        name_id = sub_components[0].split(self.minor_separator)
        sub_components = sub_components[1:]
        if name_id[0] != self.name:
            raise ValueError(f'name {name} does not match name {self.name}')
        if len(name_id) > 2:
            raise ValueError(f'name {name} is invalid because of name-ID pair')
        name_parsable = all([len(sub.split(self.minor_separator)) == 2 \
                             for sub in sub_components])
        if name_parsable == False:
            raise ValueError(f'Cannot parse name {name}. Check the name.')

    def conditionFromName(self, name, full_condition=True):
        """
        Gets the condition of the experiment (without metadata) from the name
        of a file. Discards the base name (i.e. LIA1-1).

        :param name: filename to generate condition from
        :param full_condition: Whether to return the condition with metadata + constants, or only the content of the name itself
        """
        cond = condition_from_name(name,
                minor_separator=self.minor_separator,
                major_separator=self.major_separator,
                full_condition=full_condition,
                constants=self.constants,
                metadata=self.metadata)
        return cond

    def saveMasterData(self, data_dict=None):
        """
        Saves the derived quantities in master_data to a file with the name of the experiment
        """
        if not data_dict:
            data_dict = self.data
        partial_filename = self.name
        full_filename = self.data_full_path + partial_filename + '.csv'
        with open(full_filename, 'w+') as fh:
            fh.write(str(self.constants) + '\n')
            master_data = self.master_data(data_dict)
            master_data.to_csv(fh, mode='a', index=False)


    def loadData(self, parser=parse_default):
        """
        Loads data from files located in the root data directory of the
        experiment, if available.

        :param loader: data loading function which returns a pandas DataFrame from a raw file of data. Useful if data is not already in CSV format
        """
        candidate_files = [x for x in os.listdir(self.data_full_path) \
                            if not x.startswith('.') and not x.startswith('_')
                          and x.startswith(self.name)]
        for fn in candidate_files:
            name = os.path.splitext(fn)[0]
            full_filename = self.data_full_path + fn
            data, metadata = parser(full_filename)
            self.data[name] = data
            self.metadata[name] = metadata
            self.constants = metadata # Inefficient but the best I can think of
            cond = self.conditionFromName(name)
            if self.conditions == [{}]:
                self.conditions[0] = cond
            else:
                self.conditions.append(cond)

        for val in self.metadata.values():
            self.constants = dict(self.constants.items() & val.items())

    def loadXRDData(self):
        self.loadData(parser=parse_xrd)

    def group_data(self, data=None, group_along='replicate', grouping_type='name'):
        """
        Generates unique groups for grouping data data by condition which have identical conditions except for one factor

        :param data: Dictionary of named data for which to generate groups
        :param group_along: Condition name that the data should be grouped by
        :param grouping_type: "name" or "value". Allows generation of groups into groups which have the same condition and run over one final condition, or groups which vary the value of a single condition.
        :returns groups: List of conditions without the group_along part
        """
        if data is None:
            data = self.data
        if group_along == None:
            return data
        conds = conditions_from_index(data.index)

        if grouping_type == 'value':
            grouping_names = group_along
        elif grouping_type == 'name':
            grouping_names = list(data.index.names)
            grouping_names.remove(group_along)
            if len(grouping_names) == 1:
                grouping_names = grouping_names[0]

        groups = {}
        groupby_object = data.groupby(grouping_names)
        for ind, val in groupby_object:
            if is_collection(grouping_names):
                cond = {k: v for k, v in zip(grouping_names, ind)}
            else:
                cond = {grouping_names: ind}
            name = self.nameFromCondition(cond)
            groups[name] = val

        if groups == {}:
            raise ValueError(f'No groups found for group_along={group_along}')
        return groups

    def master_data_dict(self, data=None,
            x_axis_include=[], x_axis_exclude=[],
            c_axis_include=[], c_axis_exclude=[]):
        """
        Assumes the last final value is the one to be plotted, all
        others are variables we want to plot over.

        :param data_dict: Data dictionary to generate master data dict for
        :param x_axis_include: List of factors, will include only plots which have the x-axes specified
        :param c_axis_include: List of factors, will include only plots which have curve families equal to these factors
        """
        if data is None:
            data = self.data

        master_dict = {}
        conds = self.get_conditions(data=data, full_condition=False)

        factors = factors_from_condition(conds[0])

        if len(factors) == 1: # Special case - no pairs of data
            single_table = data
            single_name = self.nameFromCondition({factors[0]: 'x'})
            return {single_name: single_table}


        factor_pairs = [x for x in permutations(factors, min(len(factors), 2))]
        if x_axis_include:
            factor_pairs = \
               [(x, c) for x,c in factor_pairs if x in x_axis_include]
        if c_axis_include:
            factor_pairs = \
               [(x, c) for x,c in factor_pairs if c in c_axis_include]
        if x_axis_exclude:
            factor_pairs = \
               [(x, c) for x,c in factor_pairs if x not in x_axis_exclude]
        if c_axis_exclude:
            factor_pairs = \
               [(x, c) for x,c in factor_pairs if c not in c_axis_exclude]

        for factor_pair in factor_pairs:
            x_factor, c_factor = factor_pair
            remaining_factors = \
                    [x for x in factors if x not in factor_pair]
            full_conditions = self.get_conditions(data)
            conditions_excluding_pairs = self.get_conditions(
                    data, exclude=factor_pair)

            if len(conditions_excluding_pairs) == 0:
                conditions_excluding_pairs = [{}]

            for cond in conditions_excluding_pairs:
                relevant_data = self.lookup(data, **cond)
                partial_cond = dict(cond, **{x_factor: 'x', c_factor: 'c'})
                curve_family_name = self.nameFromCondition(partial_cond)
                master_dict[curve_family_name] = {}

                curve_families = self.group_data(
                       relevant_data, group_along=c_factor,
                       grouping_type='value')

                for k, v in curve_families.items():
                    full_cond = dict(partial_cond,
                            **self.conditionFromName(k))
                    full_name = self.nameFromCondition(full_cond)
                    data_table = v
                    sorted_index = data_table.index.sort_values(x_factor)[0]
                    new_vals = list(sorted_index.values)
                    data_table = data_table.loc[new_vals]
                    new_index = drop_redundants(sorted_index)
                    data_table.index = new_index

                    master_dict[curve_family_name][full_name] = data_table

        return master_dict

    def lookup(self, data=None, **kwargs):
        """
        Looks up data based on a particular condition or set of conditions
        """
        if data is None:
            data = self.data
        column_truth_values = np.ones(len(data), dtype=bool)
        for k, v in kwargs.items():
            try:
                new_truth= np.array(data.index.get_level_values(k) == v)
                column_truth_values = np.logical_and(
                        column_truth_values, new_truth)
            except KeyError:
                pass

        return data.loc[column_truth_values]

    # This should be generalized into our derived quantity function. This is so close to being our derived quantity function.
    def derived_quantity(self, data=None, average_along=None,
            sum_along=None, quantity_func=None,
            averaging_type='first'):
        """
        User-facing function to average existing data along some axis

        :param along: The axis to average the data along
        :param data: Data to average
        :param averaging_type: (if Pandas DataFrame) whether to average the last column only ("last") or all columns but the first column ("first")
        """
        if data is None:
            data = self.data

        first_value = data.iloc[0, 0]

        if quantity_func is not None:
            groups = data.index
            quantity_first = True
        elif quantity_func is None:
            quantity_first = False
            if average_along is not None:
                groups = list(data.index.names)
                groups.remove(average_along)
                quantity_func = average_group
                quantity_kw = {'averaging_type': averaging_type}
            elif sum_along is not None:
                groups = list(data.index.names)
                groups.remove(sum_along)
                quantity_func = sum_group
                quantity_kw = {'summing_type': averaging_type}


        return_index_vals = []
        return_data_vals = []

        grouped_data = data.groupby(groups)
        for ind, sub_data in grouped_data:
            new_data = quantity_func(sub_data, **quantity_kw)

            return_data_vals.append(new_data)
            if is_collection(ind):
                return_index_vals.append(tuple(ind))
            else:
                return_index_vals.append((ind,))

        data_name = grouped_data.first().columns.values[0]
        return_index = pd.MultiIndex.from_tuples(
                return_index_vals,
                names=groups)
        return_index = simplify_index(return_index)
        return_values = {data_name: return_data_vals}
        derived_data = pd.DataFrame(index=return_index,
                data=return_values)

        if quantity_first == True:
            if average_along is not None:
                derived_data = average_group(derived_data,
                        average_along=average_along,
                        averaging_type=averaging_type)
            elif sum_along is not None:
                derived_data = sum_group(derived_data,
                        sum_along=sum_along,
                        summing_type=averaging_type)
            else:
                raise ValueError(f'value type you are trying to average is {type(first_value)}. Unsupported type')

        return derived_data

    def mean(self, data=None, along=None, averaging_type='first'):
        """
        User-facing function to average existing data along some axis

        :param along: The axis to average the data along
        :param data: Data to average
        :param averaging_type: (if Pandas DataFrame) whether to average the last column only ("last") or all columns but the first column ("first")
        """
        return_data = self.derived_quantity(data=data,
                average_along=along, averaging_type=averaging_type,
                quantity_func=None)
        return return_data

    def sum(self, data=None, along=None, averaging_type='first'):
        """
        User-facing function to average existing data along some axis

        :param along: The axis to average the data along
        :param data: Data to average
        :param averaging_type: (if Pandas DataFrame) whether to average the last column only ("last") or all columns but the first column ("first")
        """
        return_data = self.derived_quantity(data=data,
                sum_along=along,
                averaging_type=averaging_type,
                quantity_func=None)
        return return_data

    # This function is a mess. It needs to be split up into smaller functions. One for legend generation, one for data preparation, etc.
    # that have a clearly-delineated purpose.
    def plot(
        self, data_dict=None, along=None, sum_along=None,
        quantity_func=None, representative='',
        plotter=default_plotter, line_kw={}, subplot_kw={}, save_kw = {},
        theory_func=None, theory_kw={},
        theory_data=None, theory_exp=None,
        postfix='', x_axis_include=[], x_axis_exclude=[], c_axis_include=[], c_axis_exclude=[]):
        """
        Generates figures from loaded data.

        :param along: Averages the data along a given axis (i.e. the repilate axis)
        :param quantity_func: Quantity function to extract a quantity from a given dataset or transform that dataset
        :param fig_ax_func: Function to generate fig, ax
        :param representative: Which axis to generate a representative plot along (i.e. "replicate". Defaults to None)
        :param theory_func: Theoretical values the data should take. Assumes a function of the form y = f(x, kwargs).
        :param theory_kw: Parameters to feed into theory function. Assumed to be a single dict.
        :param theory_data: Theoretical data as a Pandas DataFrame to be plotted along with the data.
        :param subplot_kw: Keyword arguments to be passed to fig.subplots()
        :param save_kw: Keyword arguments to be passed to fig.savefig()
        :param x_axis_include: List of factors, will include only plots which have the x-axes specified
        :param x_axis_exclude: List of factors for which you do not want to be plotted on the x-axis
        :param c_axis_include: Complete list of factors for which you want to generate c-axis plots
        :param c_axis_exclude: Complete list of factors for which you do not want to be plotted on the c-axis
        """
        plotted_figs = []
        plotted_axes = []
        if along is not None:
            if postfix != '': postfix += self.major_separator
            postfix += 'averaged'
        if sum_along is not None:
            if postfix != '': postfix += self.major_separator
            postfix += 'summed'

        if data_dict is None:
            data_dict = self.data

        if representative:
            data_dict = self.group_data(data_dict, group_along=representative)
            for k in data_dict.keys():
                data_dict[k] = list(data_dict[k].values())[0]
            if postfix != '': postfix += self.major_separator
            postfix += 'representative'

        if quantity_func:
            dict_to_plot = self.derived_quantity(
                quantity_func=quantity_func,
                data_dict=data_dict,
                along=along,
                sum_along=sum_along)
        else:
            dict_to_plot = data_dict

        first_value = list(dict_to_plot.values())[0]
        data_is_scalar = is_scalar(first_value)
        if data_is_scalar:
            # Generate a bunch of dictionaries with the appropriate
            # names from this mmaster table so we can plot them.
            dict_to_plot = self.master_data_dict(
                    dict_to_plot, x_axis_include=x_axis_include,
                    x_axis_exclude=x_axis_exclude,
                        c_axis_include=c_axis_include,
                        c_axis_exclude=c_axis_exclude)
            if dict_to_plot == {}:
                raise ValueError('ERROR: No plots generated for combination of desired inclusion / exclusion criteria.')

        for name, data in dict_to_plot.items():
            is_pandas = isinstance(data, pd.DataFrame)
            is_dict = isinstance(data, dict)
            if is_dict:
                fig, ax = None, None
                legend = []
                outer_cond = self.conditionFromName(name)
                c_factor = ''
                for k, v in outer_cond.items():
                    if v == 'c':
                        c_factor = k

                for inner_name, inner_data in data.items():
                    if theory_exp is not None:
                        theory_data = match_theory_data(inner_name, theory_exp)
                    if not fig:
                        fig, ax = plotter(
                            inner_data, theory_func=theory_func,
                            theory_kw=theory_kw,
                            theory_data=theory_data,
                            subplot_kw=subplot_kw,
                            line_kw=line_kw)
                    else:
                        plotter(
                            inner_data, fig=fig, ax=ax,
                            theory_func=theory_func, theory_kw=theory_kw,
                            theory_data=theory_data,
                            subplot_kw=subplot_kw, line_kw=line_kw)

                    # Generate the legend
                    inner_cond = self.conditionFromName(inner_name)
                    stripped_inner_cond = \
                        {k: v for k, v in inner_cond.items() if k==c_factor}
                    stripped_inner_name = self.nameFromCondition(stripped_inner_cond)

                    if theory_func is None and theory_data is None:
                        legend.append(self.prettify_name(stripped_inner_name))
                    else:
                        legend.append(self.prettify_name(stripped_inner_name) + ' (Measured)')
                        legend.append(self.prettify_name(stripped_inner_name) + ' (Theory)')
                ax.legend(legend, fontsize=10)
                plotted_figs.append(fig)
                plotted_axes.append(ax)
            elif is_pandas:
                fig, ax = plotter(data, line_kw=line_kw,
                        theory_func=theory_func, theory_kw=theory_kw,
                        theory_data=theory_data,
                        subplot_kw=subplot_kw)
                plotted_figs.append(fig)
                plotted_axes.append(ax)

            extension = '.png'
            if 'format' in save_kw:
                extension = '.' + save_kw['format']
            if postfix != '':
                full_filename = self.figures_full_path + name + \
                        self.major_separator + postfix + extension
            else:
                full_filename = self.figures_full_path + name + extension

            prettifyPlot(fig=fig,ax=ax)
            fig.savefig(full_filename, bbox_inches='tight')
        return plotted_figs, plotted_axes

    def plotPSD(
            self, along=None, representative=False, **kwargs):
        def psdFunction(data, cond):
            return power_spectrum(data, **kwargs)
        self.plot(
            along=along,
            quantity_func=psdFunction, plotter=power_spectrum_plot,
            representative=representative, postfix='PSD')

    def process(self):
        """
        Execute the script with the same name located in the processing code directory.
        """
        file_contents = open(self.process_full_path + self.name + '.py').read()
        exec(file_contents)

    def process_photocurrent(
            self, reference_condition, along=None, representative=False, sim_exp=None):
        """
        Generates derived quantities for photocurrent given measured
        voltages, system gain, sampling frequency, etc.

        :param reference_condition: The condition which contains the reflectance reference. Must be unique.
        """
        if sim_exp is None:
            sim_exp = Experiment(name='REFL2', kind='simulation')
            sim_exp.loadData()

        potential_reference_data = \
            list(sim_exp.lookup(material='Au').values())
        if len(potential_reference_data) == 0:
            raise ValueError(f'No reference data found for the specified condition: {reference_condition}. Available data names are {sim_exp.data.keys()}')
        elif len(potential_reference_data) > 1:
            raise ValueError(f'Reference condition is not unique. Specify a unique reference condition.')
        elif len(potential_reference_data) == 1:
            theoretical_Au_R0 = potential_reference_data[0]

        reference_photocurrent = self.lookup(**reference_condition)
        reference_photocurrent_dc = self.derived_quantity(
                quantity_func=dc_photocurrent,
                data_dict=reference_photocurrent)
        reference_photocurrent_table = self.master_data(
                reference_photocurrent_dc)

        dc_photocurrents_dict = self.derived_quantity(
                quantity_func=dc_photocurrent,
                data_dict=self.data)
        dc_photocurrents = self.master_data(dc_photocurrents_dict)

        mod_photocurrents_dict = self.derived_quantity(
                quantity_func=modulated_photocurrent,
                data_dict=self.data)
        mod_photocurrents = self.master_data(mod_photocurrents_dict)

        R0_table = normalize_reflectance(
                dc_photocurrents,
                reference_photocurrent_table,
                theoretical_Au_R0,
                column_units=ureg.nm, target_units=ureg.nA)
        dR_table = normalize_reflectance(
                mod_photocurrents, reference_photocurrent_table,
                theoretical_Au_R0,
                column_units=ureg.nm, target_units=ureg.nA)

        # We need to explicitly add a "spectra" to the names for dR and R0.
        R0_dict = self.data_from_master(R0_table)
        R0_dict = {k + '~spectra=R0': v for k, v in R0_dict.items()}
        dR_dict = self.data_from_master(dR_table)
        dR_dict = {k + '~spectra=dR': v for k, v in dR_dict.items()}

        noise_photocurrents_dict = self.derived_quantity(
                quantity_func=noise_current,
                data_dict=self.data,
                quantity_kw={'filter_cutoff': 200*ureg.Hz})

        return (R0_dict, dR_dict, noise_photocurrents_dict)

    def plot_photocurrent(
            self, reference_condition, along=None,
            representative=False, sim_exp=None,
            x_axis_include=['wavelength'],
            c_axis_include=[], x_axis_exclude=[], c_axis_exclude=[]):
        """
        Generates photocurrent plots

        """
        if sim_exp is None:
            sim_exp = Experiment(name='REFL2', kind='simulation')
            sim_exp.loadData()

        R0_dict, dR_dict, noise_photocurrents_dict = \
                 self.process_photocurrent(
                         reference_condition=reference_condition,
                         along=along,
                         representative=representative, sim_exp=sim_exp)

        dR_figs, dR_axes = self.plot(
            data_dict=dR_dict,
            theory_exp=sim_exp,
            x_axis_include=x_axis_include,
            x_axis_exclude=x_axis_exclude,
            c_axis_include=c_axis_include,
            c_axis_exclude=c_axis_exclude,
            subplot_kw={
                'ylabel': r'$\Delta R_{amp}$',
                'xlim': (870, 1100),
                'ylim': (-4e-4, 4e-4),
                },
            postfix='dR')
        R0_figs, R0_axes = self.plot(
                data_dict=R0_dict,
                theory_exp=sim_exp,
                x_axis_include=x_axis_include,
                x_axis_exclude=x_axis_exclude,
                c_axis_include=c_axis_include,
                c_axis_exclude=c_axis_exclude,
                subplot_kw={
                    'ylabel': r'$R_0$',
                    'xlim': (870, 1100),
                    'ylim': (0, 1),
                    },
                postfix='R0')
        inoise_figs, inoise_axes = self.plot(
                data_dict=noise_photocurrents_dict,
                theory_func=inoise_func_dBAHz,
                x_axis_include=x_axis_include,
                x_axis_exclude=x_axis_exclude,
                c_axis_include=c_axis_include,
                c_axis_exclude=c_axis_exclude,
                plotter=power_spectrum_plot,
                postfix='inoise')

        return (R0_figs, R0_axes,
                dR_figs, dR_axes,
                inoise_figs, inoise_axes)
