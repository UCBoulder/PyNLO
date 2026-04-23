PyNLO: Python Nonlinear Optics
=============================

This is a fork of the original PyNLO, a package for modeling the nonlinear interaction of light with matter. It started as an attempt to add 2nd-order nonlinearities to the pulse propagation model and grew into a large-scale rewrite. It is not yet at feature parity with the original, but it is getting close! Contributions and suggestions are welcome.

Complete documentation for PyNLO can be found on the repository's GitHub Pages website.

Introduction
------------

The PyNLO package provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with matter. It provides many functionalities for representing pulses of light and nonlinear materials.

Features:

- A solver for the propagation of light through materials with both 2nd- and 3rd-order nonlinearities.
- A highly-efficient adaptive step size algorithm based on the ERK4(3)-IP method from `Balac and Mahé (2013) <https://doi.org/10.1016/j.cpc.2012.12.020>`_.
- A flexible object-oriented system for treating laser pulses and optical modes.
- ...and much more!

Installation
------------

The FFT backend `FFTW3 <https://www.fftw.org/>`_ makes this version of PyNLO **cross-platform**.

Installing to Use
~~~~~~~~~~~~~~~~~

Installing with ``conda``
^^^^^^^^^^^^^^^^^^^^^^^^^

PyNLO is designed to work most easily with the `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution. For a functional installation with ``conda``, run::

    git clone https://github.com/UCBoulder/PyNLO.git
    cd pynlo
    conda activate base
    conda install -c conda-forge unidep=3
    unidep install -n pynlo .
    conda activate pynlo

This command clones the repo to your machine and installs it in a new ``conda`` environment named ``pynlo``. Test out your installation with the scripts in the examples folder.

Installing with ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

PyNLO is designed to use with ``conda`` but uses `unidep <https://unidep.readthedocs.io/en/latest/index.html>`_ for packaging, which has compatibility with ``pip``. However, PyNLO relies on `FFTW3 <https://www.fftw.org/>`_, which does not ship with the ``pip`` distribution of ``pyfftw`` by default. This must be installed separately, so using ``conda`` (which has ``FFTW3`` packaged with it) is highly recommended.

Installing to Develop
~~~~~~~~~~~~~~~~~~~~~

If you're looking to make changes to the source code of ``pynlo``, add the ``-e`` flag so that changes made on the local version of the repository are reflected in your local scripts::

    git clone https://github.com/UCBoulder/PyNLO.git
    cd pynlo
    conda activate base
    conda install -c conda-forge unidep=3
    unidep install -n pynlo -e .
    conda activate pynlo

Contributing
------------

Open an issue or discussion on the GitHub repository to add suggestions for improvement, ask questions, or make other comments. Additions to the tests, examples, and documentation are highly appreciated. New contributions should be based off the ``develop`` branch.

License
-------

PyNLO is licensed under the `GNU LGPLv3 license <https://choosealicense.com/licenses/lgpl-3.0/>`_. This means that you are free to use PyNLO for any project, but all modifications to it must be kept open source. PyNLO is provided "as is" with absolutely no warranty.