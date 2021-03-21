import itertools
import numpy as np
import pandas as pd
import os
import re
from io import BytesIO
from pathlib import Path
from matplotlib.figure import Figure
from sugarplot import plt, prettifyPlot, default_plotter
from spectralpy import power_spectrum, title_to_quantity
from spectralpy import to_standard_quantity, powerSpectrumPlot
from sciparse import parse_xrd, parse_default
from itertools import permutations

class Experiment:
    """
    :param kwargs: A list of keyword arguments to pass into the experiment. These can be scalars/strings, or lists. If lists, all cartesian products of all possible experiments will be run.
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
        self.minor_separator = '-'
        self.name = name
        self.ident = ident
        self.base_path = base_path
        self.data = {}
        self.metadata = {}
        self.verbose = verbose
        self.measure_func = measure_func

        if kind == 'designs' or kind == 'design':
            self.data_full_path = base_path + '/designs/'+ name + '/data/'
            self.figures_full_path = base_path + '/designs/' + name + '/figures/'
        else:
            self.data_full_path = self.base_path + '/data/' + kind + '/' + name + '/'
            self.figures_full_path = self.base_path + '/figures/' + kind + '/' + name + '/'
        self.factors = {k:v for k, v in kwargs.items() \
             if isinstance(v, (list, np.ndarray))}
        self.constants = {k:v for k, v in kwargs.items() \
                 if not isinstance(v, (list, np.ndarray))}

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

    def generate_conditions(self, comb_type='cartesian', **factors):
        """
        Generates a list of desired conditions from the specified factors and their levels.

        :param factors: dictionary with the factors and their desired levels
        :param comb_type: "cartesian" (outer product) - generates a full factorial experiment from the set of factors, "individual" creates an individual condition from the specified factor levels, "1to1" generates a 1-to-1 mapping of factor levels.
        """
        if not factors:
            factors = self.factors
        cond_list = []
        keys = factors.keys()
        vals = factors.values()
        if comb_type == 'cartesian':
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

    def append_condition(self, **kwargs):
        """
        Appends a single condition to the existing list of conditions using the metadata already defined in the experiment initialization. Modifies in-place.

        :param kwargs: Keyword arguments with the values for the condition you want to append
        """
        extra_cond = self.generate_conditions(
                comb_type='individual', **kwargs)
        self.conditions.extend(extra_cond)

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

    def factors_from_condition(self, cond):
        factors = [x for x in cond.keys()]
        return factors

    def get_conditions(self, data_dict=None, exclude=[]):
        """
        Get all the conditions from a given set of data

        :param data_dict: The data you wish to extract conditions from
        :param exclude: The list of factors you would like to exclude from consideration.
        :param include: The complete list of factors you want included
        :returns conds: A list of conditions that match the inclusion and exclusion criteria
        """
        if not data_dict:
            data_dict = self.data
        if isinstance(exclude, str):
            exclude = [exclude]
        conds = [self.conditionFromName(name, full_condition=False) \
            for name in data_dict.keys()]

        for cond in conds:
            for factor in exclude:
                cond.pop(factor, None) # Remove the condition

        return_conds = []
        [return_conds.append(c) for c in conds if c not in return_conds]
        return return_conds

    def nameFromCondition(self, condition):
        """
        Generates filename for an experiment from a condition, returning a name
        with the metadata truncated, experiment name included

        :param condition: Condition to generate name for
        """
        partial_filename = self.name
        if 'ident' in condition.keys():
            partial_filename = partial_filename + self.minor_separator + \
                    condition['ident']
            condition.pop('ident')
        for key, val in condition.items():
            if key not in self.constants.keys() and key not in \
            self.metadata.keys():
                partial_filename += self.major_separator + key + \
                self.minor_separator + str(val)

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
        """
        sub_components = name.split(self.major_separator)
        name_id = sub_components[0].split(self.minor_separator)
        sub_components = sub_components[1:]
        cond = {}
        self.validateName(name)
        if len(name_id) > 1: cond.update({'ident': name_id[1]})

        cond.update({sub.split(self.minor_separator)[0]:sub.split(self.minor_separator)[1] \
                for sub in sub_components})
        for key, val in cond.items():
            try: cond[key] = int(val)
            except ValueError:
                try: cond[key] = float(val)
                except ValueError: cond[key] = val
        if full_condition:
            cond.update(self.constants)
            if name in self.metadata.keys():
                cond.update(self.metadata[name])
        return cond

    def saveRawResults(self, raw_data, condition):
        """
        Saves raw results from the experiment performed with a given condition

        :param condition: Experimental condition as a dictionary
        :param raw_data: Raw data as a pandas array to save
        """
        partial_filename = self.nameFromCondition(condition)
        self.data[partial_filename] = raw_data
        full_filename = self.data_full_path + partial_filename + '.csv'
        with open(full_filename, 'w+') as fh:
            fh.write(str(self.constants) + '\n')
            raw_data.to_csv(fh, mode='a', index=False)

    def derived_quantity(self, data_dict, quantity_func,
                               average_along=None, **kwargs):
        """
        Extracts derived quantities from a named dictionary of data with some
        arbitrary input functioin, and optional averaging along an arbitrary
        axis.

        :param data_dict: Named data dictionary from which to extract derived quantities
        :param quantity_func: Function to apply to generate data
        :param average_along: Axis to average along (i.e. replicate or None)
        """
        derived_dict = {}
        is_scalar = False
        for name, data in data_dict.items():
            cond = self.conditionFromName(name)
            quantity = quantity_func(data, cond, **kwargs)
            derived_dict[name] = quantity
            if isinstance(quantity, float): is_scalar = True

        if average_along != None:
            derived_dict = self.average_data(
                data_dict=derived_dict, average_along=average_along)
        return derived_dict

    def saveMasterData(self, data_dict=None):
        """
        Saves the derived quantities in master_data to a file with the name of the experiment
        """
        if not data_dict: data_dict = self.data
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

        for val in self.metadata.values():
            self.constants = dict(self.constants.items() & val.items())

    def loadXRDData(self):
        self.loadData(parser=parse_xrd)

    def generate_groups(self, data_dict=None, group_along='replicate', grouping_type='name'):
        """
        Generates unique groups for grouping data data by condition which have different replicates but the some condition otherwise

        :param data_dict: Dictionary of named data for which to generate groups
        :param group_along: Condition name that the data should be grouped by
        :param grouping_type: "name" or "value". Allows generation of groups into groups which have the same condition and run over one final condition, or groups which vary the value of a single condition.
        :returns groups: List of conditions without the group_along part
        """
        if data_dict == None:
            data_dict = self.data
        if group_along == None:
            return data_dict
        names = [name for name in data_dict.keys()]
        conds = [self.conditionFromName(name, full_condition=False) \
                for name in names]
        groups = {}
        for cond, cond_name in zip(conds, names):
            if group_along in cond.keys():

                if grouping_type == 'name':
                    group_conditions = {k: cond[k] for k in cond.keys() if k != group_along}
                elif grouping_type == 'value':
                    group_conditions = {group_along: cond[group_along]}
                else:
                    raise ValueError(f'grouping_type must be "name" or "value". Found {grouping_type}')

                group_name = self.nameFromCondition(group_conditions)
                if group_name not in groups.keys():
                    groups[group_name] = {}
                groups[group_name][cond_name] = None

        if groups == {}:
            raise ValueError(f'No groups found for group_along={group_along}')
        return groups

    def group_data(self, data_dict=None, group_along=None,
            grouping_type='name'):
        """
        Groups data by name (i.e. fix the condition and vary one of the values in that condition) or by value (fix a single factor of the condition and vary its values, including all other combinations).

        """
        if data_dict == None:
            data_dict = self.data
        if group_along == None:
            return data_dict
        groups = self.generate_groups(
                data_dict=data_dict, group_along=group_along,
                grouping_type=grouping_type)
        grouped_data = groups.copy()

        for group_name, group in grouped_data.items():
            for name in group.keys():
                grouped_data[group_name][name] = data_dict[name]

        return grouped_data

    def master_data(self, data_dict=None, value_name='Value'):
        """
        NOTE: Currently only designed for data_dicts with scalar values. Not design for data frames or multi-valued datasets. I need to fix this.

        """
        if data_dict is None: data_dict = self.data
        return_frame = pd.DataFrame()

        for name in data_dict.keys():
            conds = self.conditionFromName(name, full_condition=False)
            new_row = pd.DataFrame()
            for k, v in conds.items():
                new_row[k] = [v]
            new_row[value_name] = data_dict[name]
            new_row.index = [len(return_frame)]
            return_frame = return_frame.append(new_row)

        for col in return_frame.columns.values:
            if all(return_frame[col][0] == return_frame[col]):
                return_frame = return_frame.drop(col, axis=1)

        return return_frame

    def data_from_master(self, master_data):
        """
        :param master_data: Master dataset as a pandas dataframe
        """
        if not isinstance(master_data, pd.DataFrame):
             raise ValueError(f"Unspported data type {type(master_data)}. Can only support pandas DataFrame")

        col_names = master_data.columns.values[:-1]

        data_last = master_data.iloc[:,-1]
        data_without_last = master_data.iloc[:,:-1]

        data_dict = {}
        for tuple_row, (ind, regular_row) in \
            zip(master_data.itertuples(index=False), master_data.iterrows()):

            reduced_names = regular_row.index[:-1]
            reduced_values, last_val = \
                list(tuple_row)[:-1], list(tuple_row)[-1]
            cond = {cname: val for cname, val in \
                zip(reduced_names, reduced_values)}
            name = self.nameFromCondition(cond)
            data_dict[name] = last_val

        return data_dict

    def master_data_dict(self, data_dict=None):
        """
        Assumes the last final value is the one to be plotted, all
        others are variables we want to plot over.
        """
        if not data_dict:
            data_dict = self.data

        master_dict = {}
        conds = [self.nameToCondition(x, full_condition=False) \
            for x in data_dict.keys()]
        factors = self.factors_from_condition(conds[0])

        if len(factors) == 1: # Special case - no pairs of data
            single_table = self.master_data(data_dict)
            single_name = self.nameFromCondition({factors[0]: 'all'})
            return {single_name: single_table}


        factor_pairs = [x for x in permutations(factors, min(len(factors), 2))]
        for factor_pair in factor_pairs:
            x_factor, c_factor = factor_pair
            remaining_factors = \
                    [x for x in factors if x not in factor_pair]
            conditions_excluding_pairs = self.get_conditions(
                    data_dict, exclude=factor_pair)

            if len(conditions_excluding_pairs) == 0:
                conditions_excluding_pairs = [{}]

            for cond in conditions_excluding_pairs:
                remaining_data = self.lookup(data_dict, **cond)
                full_cond = dict({x_factor: 'all'}, **cond)
                curve_family_name = self.nameFromCondition(full_cond)
                master_dict[curve_family_name] = {}

                curve_families = self.group_data(
                       remaining_data, group_along=c_factor,
                       grouping_type='value')
                for k, v in curve_families.items():
                    data_table = self.master_data(v)
                    master_dict[curve_family_name][k] = data_table

        return master_dict

    def lookup(self, data_dict=None, **kwargs):
        """
        Looks up data based on a particular condition or set of conditions
        """
        if data_dict==None:
            data_dict = self.data
        return_dict = {}
        for name, val in data_dict.items():
            cond = self.conditionFromName(name)
            contains_required_keys = \
                all([desired_cond in cond.keys() \
                     for desired_cond in kwargs.keys()])

            if contains_required_keys:
                contains_required_values = \
                        all([cond[k] == v for k, v in kwargs.items()])
                if contains_required_values:
                    return_dict[name] = val

        return return_dict

    def average_data(self, data_dict=None, average_along=None, averaging_type='first'):
        """
        User-facing function to average existing data along some axis

        :param data_dict: Data dict to average
        :param average_along: The axis to average the data along
        :param averaging_type: (if Pandas DataFrame) whether to average the last column only ("last") or all columns but the first column ("first")
        """
        if data_dict == None:
            data_dict = self.data
        grouped_data = self.group_data(
            data_dict=data_dict, group_along=average_along,
            grouping_type='name')
        averaged_data = {}

        for group_name, group in grouped_data.items():
            first_name  = list(group.keys())[0]
            first_item = list(group.values())[0]
            is_pandas = isinstance(first_item, pd.DataFrame)
            is_scalar = isinstance(first_item, (int, float))
            N_items = len(group)

            if is_pandas:
                averaged_data[group_name] = first_item.copy()
                if averaging_type == 'last':
                    for data in group.values():
                        averaged_data[group_name].iloc[:,-1] += data.iloc[:,-1]
                    averaged_data[group_name].iloc[:,-1] -= first_item.iloc[:,-1]
                    averaged_data[group_name].iloc[:,-1] /= N_items

                elif averaging_type == 'first':
                    for data in group.values():
                        averaged_data[group_name].iloc[:,1:] += data.iloc[:, 1:]
                    averaged_data[group_name].iloc[:,1:] -= first_item.iloc[:,1:]
                    averaged_data[group_name].iloc[:,1:] /= N_items
                else:
                     raise ValueError(f'Averaging type {averaging_type} not recognized. Available types are "first" and "last"')

            elif is_scalar:
                averaged_data[group_name] = first_item
                for data in group.values():
                    averaged_data[group_name] += data
                averaged_data[group_name] -= first_item
                averaged_data[group_name] /= N_items
            else:
                raise ValueError(f'type {type(first_item)} not supported. Available types are scalar, Pandas Array')

        return averaged_data

    # This function is a mess. It needs to be split up into smaller functions
    # that have a clearly-delineated purpose.
    def plot(
        self, data_dict=None, average_along=None,
        quantity_func=None, quantity_name = 'Value', representative='',
        plotter=default_plotter, theory_func=None, theory_param_dict=None,
        postfix='', **kwargs):
        """
        Generates figures from loaded data. Currently assumes the data is in
        the form of a pandas array.

        :param average_along: Averages the data along a given axis (i.e. the repilate axis)
        :param quantity_func: Quantity function to extract a quantity from a given dataset or transform that dataset
        :param fig_ax_func: Function to generate fig, ax
        :param representative: Which axis to generate a representative plot along (i.e. "replicate". Defaults to None)
        :param theory_func: Theoretical values the data should take. Assumes a function of the form y = f(x, kwargs).
        :param theory_param_dict: Parameters to feed into theory function. Should be a dictionary with the same names as the data_dict
        """
        if average_along:
            if postfix != '': postfix += self.major_separator
            postfix += 'averaged'
        if not data_dict:
            data_dict = self.data
        if representative:
            data_dict = self.group_data(data_dict, group_along=representative)
            for k in data_dict.keys():
                data_dict[k] = list(data_dict[k].values())[0]
            if postfix != '': postfix += self.major_separator
            postfix += 'representative'

        if quantity_func:
            dict_to_plot = self.derived_quantity(
                data_dict, average_along=average_along,
                quantity_func=quantity_func, **kwargs)
            random_value = list(dict_to_plot.values())[0]
            if isinstance(random_value, float):
                # Generate a bunch of dictionaries with the appropriate
                # names from this mmaster table so we can plot them.
                master_data = self.master_data(dict_to_plot,
                                              value_name=quantity_name)
                dict_to_plot = self.master_data_dict(master_data)
        else:
            dict_to_plot = data_dict

        for name, data in dict_to_plot.items():
            is_pandas = isinstance(data, pd.DataFrame)
            is_dict = isinstance(data, dict)
            fig, ax = plotter(data, **kwargs)
            if is_dict:
                # This is where we finally get to do iterative
                legend = []
                for inner_name, inner_data in data.items():
                    default_ploter(
                        data, fig=fig, ax=ax, theory_func=theory_func,
                        theory_kw=theory_kw, **kwargs)
                    legend.append(inner_name)
                ax.legend(legend)

            prettifyPlot(fig=fig,ax=ax)
            if postfix != '':
                full_filename = self.figures_full_path + name + \
                        self.major_separator + postfix + '.png'
            else:
                full_filename = self.figures_full_path + name + '.png'
            fig.savefig(full_filename)

    def plotPSD(self, average_along=None,
                           representative=False, **kwargs):
        def psdFunction(data, cond):
            return power_spectrum(data, **kwargs)
        self.plot(
            average_along=average_along,
            quantity_func=psdFunction, plotter=powerSpectrumPlot,
            representative=representative, postfix='PSD', **kwargs)
