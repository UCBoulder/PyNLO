# PyNLO: Python Nonlinear Optics
This is a fork of the original PyNLO, a package for modeling the nonlinear interaction of light with matter. It started as an attempt to add 2nd-order nonlinearities to the pulse propagation model and grew into a large-scale rewrite. It is not yet at feature parity with the original, but it is getting close! Contributions and suggestions are welcome.

Complete documentation for PyNLO can be found on the repository's GitHub Pages website.

## Introduction
The PyNLO package provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with matter. It provides many functionalities for representing pulses of light and nonlinear materials.

Features:

- A solver for the propagation of light through materials with both 2nd- and 3rd-order nonlinearities.

- A highly-efficient adaptive step size algorithm based on the ERK4(3)-IP method from [Balac and Mahé (2013)](https://doi.org/10.1016/j.cpc.2012.12.020).

- A flexible object-oriented system for treating laser pulses and optical modes.

- ...and much more!

## Installation
### Installing to Use
PyNLO requires Python 3. If you do not already have Python, the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution is a good place to start. With python installed, run,

    pip install git+https://github.com/UCBoulder/PyNLO.git

to install `pynlo` directly from this repository. Test out your installation with the scripts in the examples folder.

Add the ``--no-deps`` option if in a conda environment, in which case, you should separately install the dependencies using the ``conda install`` command. PyNLO depends on the `numpy`, `scipy`, `numba`, and `mkl_fft` packages. The `matplotlib` package is necessary to view real-time simulation updates and to run the example code.
### Installing to Develop
If you're looking to make changes to the source code of `pynlo`, it is highly recommeded that you clone the repository with `git` and install as an editable package with,
	pip install -e <path/to/pynlo>

# Contributing

Open an issue or discussion on the GitHub repository to add suggestions for improvement, ask questions, or make other comments. Additions to the tests, examples, and documentation are highly appreciated. New contributions should be based off the `develop` branch.


# License
-------
PyNLO is licensed under the [GNU LGPLv3 license](https://choosealicense.com/licenses/lgpl-3.0/). This means that you are free to use PyNLO for any project, but all modifications to it must be kept open source. PyNLO is provided "as is" with absolutely no warranty.
