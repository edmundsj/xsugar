# xsugar  Experimental Scripting Framework
[![Build](https://github.com/edmundsj/xsugar/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/edmundsj/xsugar/actions/workflows/python-package-conda.yml) [![docs](https://github.com/edmundsj/xsugar/actions/workflows/build-docs.yml/badge.svg)](https://github.com/edmundsj/xsugar/actions/workflows/build-docs.yml) [![codecov](https://codecov.io/gh/edmundsj/xsugar/branch/main/graph/badge.svg?token=yDhjXn4fZh)](https://codecov.io/gh/edmundsj/xsugar)

Often in my Ph.D. I found myself doing the same thing over and over again: writing a script to execute an experiment, and writing a script to parse, analyze, and plot the data. These scripts would always be saving data and figures to roughly the same place, and the general process carried out was the same. I came to realize if the data is numerical and tabular, and the experimental design relies on a finite number of conditions, much of this process can (and should) be automated. This package is my attempt to do that.

## Features
- Full factorial experiment generation for N factors
- Automation of data parsing, saving, with support for arbitrary numerical datasets
- Intelligent data plotting and figure generation

## Documentation
Documentation can be found on [github.io](https://edmundsj.github.io/xsugar/)

## Getting Started

### Installing the python module
Soon, this software should be pip-installable, and the following should work:

```
pip install xsugar
```

## Simple Example
In this experiment we gonig change two factors: the wavelength of an optical source and the temperature. We are going to measure the intensity (which I've just fudged to be a sinewave). 

```
from xsugar import Experiment
import numpy as np
import pandas as pd

wavelength = [500, 600, 700]
temperature = [25, 35]
sampling_frequency = 10000
measurement_time = 0.01

def measure_output(cond):
    time = np.arange(0, 1 / cond['sampling_frequency'], cond['measurement_time'])
    data = np.sin(time)
    data = pd.DataFrame({'Time (ms)': time, 'Current (nA)': data})
    return data

exp = Experiment(name='REFL1', kind='photocurrent', base_path='/home/jordan/experimental_directory/', measure_func=measure_output, wavelength=wavelength, temperature=temperaturesampling_frequency=sampling_frequency)
exp.Execute()
exp.plot()
        
```

## Terminology and Architecture
This module is designed to run **experiments**, which consist of a number experimental **conditions** in which the experimenter manipulates zero or more **factors**. These conditions also have associated **metadata**. 

In this module I use the term **factors** to refer to an experimental variable that is deliberately manipulated by the experimenter<sup>1</sup>. Each unique combination of factors used in an experiment I refer to as a **condition**. If you wish to run replicates with identical factors, the replicate number is considered (for the purposes of this module) a factor, and part of an experimental condition. Each experimental **condition** also has a set of **metadata** associated with it. This might be unique to the condition (i.e. with the time and date the experiment was run), or it might be common to all conditions (i.e. the sampling frequency used to aquire data). 

Each experiment is assumed to have a unique **name**, which should not contain any hyphens (-) or tildes (~). All other characters are permitted. This name should be unique to the experiment and easy to recall (i.e. CVDDOP1 or HIPPONEURON1). The experiment may also have an optional **identifier** (i.e. 1), where many similar experiments are being done. Experiments are assumed to also have a **kind**, which is just an additional way of categorizing experiments.

All experimental data exists inside a single **base directory**. Inside this **base directory**, this module will create (if it does not already exist) a folder called **data/**, and a folder called **figures/**. Inside each of these folders, there will be subfolders for each experimental **kind**, and within those folders, folders with each unique experimental **name**, where the data or figures will be stored.

Each experiment involves the measurement of some quantity, via a **measurement function**. This function should take as a single argument, a dictionary which contains the full experimental condition (the levels of all the factors, plus any associated metadata), and it should return the measured data, which should be in the form of a scalar, a numpy array, or a pandas DataFrame. The data returned by a measurement function will be saved in a filename that combines the factors and their current levels with the experimental name using tildes and underscores (i.e. if the name is "TEST1", kind is "photocurrent", and the factors are wavelength, which is currently at a value of 2, and temperature, at a value of 25, the filename will be TEST1~wavelength-2~temperature-25.csv, and will be saved in the "photocurrent" directory within the base directory.). 

After the data is measured, you may want to extract a quantity from each of your datasets (for example, the mean value of measured photocurrent). You can do this with **quantity functions**. These functions should take two arguments: the data and the experimental condition. They may also take any number of additional keyword arguments as desired, for example, to parameterize curve fitting. These quantity functions can be used directly or passed in as an argument to the plotting / analysis methods. When extracting desired quantities, you can take the average result (along a given factor), or representative result. These quantities may be scalars, or they may transform the data (such as extracting a power spectral density).

Finally, you may want to generate figures from the raw data, or from the derived quantities. Plotting the raw data is the default behavior, but if you supply a quantity function to the plotting function, it will generate a set of plot families for that derived quantity. If the derived quantity is a dataset itself, the plotter will just generate one plot for condition. If the derived quantity is a scalar, the behavior is more complex.

If the derived quantity is a scalar (for example full-width-half-max for a curve fitting procedure on a dataset) the default method of figure generation is to combinatorially generate a set of all possible plot families. For example, if two factors are involved, there will be two plots. One plot will have factor 1 on the x-axis, and several curves which correspond to different levels of factor 2. The other plot will have factor 2 on the x-axis, and different curves which correspond to varying levels of factor 1.

If there are three factors, this procedure is repeated. First factor 1 is placed on the x-axis, and the number of figures generated this will be the number of levels of factor 2 plus the number of levels of factor 3. Then factor 2 will be placed on the x-axis, and so on. This plotting scheme is designed to work with N factors, each of which have any number of levels. This can generate a large number of plots, so plots with a given x-axis can be excluded or included specifically with arguments to the plotting function.

<sup>1</sup> I decided not to use "variable" because of how overloaded it is in the context of programming. 

