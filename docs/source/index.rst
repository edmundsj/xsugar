.. Experiments documentation master file, created by
   sphinx-quickstart on Sat Feb 27 10:28:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Experiments's documentation!
=======================================
Introduction, Background
---------------------------
The purpose of this module is to automate experimentation through scripting. I
generate large amounts of data with the experiments I do, and I found myself
frustrated by the need to constantly update my experimental scripts to process
the data and generate figures from it each time I added some more data. 

This module assumes that your data is structured in a single ``data``
directory, with two levels of hierarchy: the first level for the type of
experiment you are doing (i.e. ``photocurrent`` for a photocurrent measurement)
and the second level with a unique experimental namefor the specific experiment you are running (i.e.
``REFL1``). 

Getting Started
-----------------


API
-------
.. autoclass:: experiments.Experiment
    :members:
    :undoc-members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
