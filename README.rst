.. image:: https://zenodo.org/badge/782623936.svg
  :target: https://doi.org/10.5281/zenodo.15701630

process-fof
-----------
Post processing scripts for AREPO friends-of-friends output. 


Overview:
---------
process-fof is a set of python scripts to post-process AREPO FOF runs. Original paper: `Williams et al (2025)`.
Public version is still somewhat in development. Reach out to Claire (clairewilliams@astro.ucla.edu) for help. 

Getting Started
===============

This section will guide you through the steps to set up and start using the repository.

Prerequisites
-------------
Before you begin, ensure you have the following installed:

- Python 3.x
- Git

Clone the Repository
--------------------
First, clone the repository to your machine using Git:

.. code-block:: bash

    git clone https://github.com/astro-claire/process-fof.git
    cd process-fof

.. Create a Virtual Environment
.. ----------------------------
.. It is recommended to use a virtual environment to manage dependencies. Create and activate a virtual environment:

.. .. code-block:: bash

..     python -m venv venv
..     source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies
--------------------
Install the required dependencies using `pip`:

.. code-block:: bash

    pip install -r requirements.txt

Configuration
-------------
You will need two configuration files to run the scripts. ``config.yaml`` provides information about the simulation settings and directories for I/O. ``constants.yaml`` contains the cgs values of the units used in your simulation. Copy the example configuration files and edit with your settings:

.. code-block:: bash

    cp examples/example_config.yaml config/config.yaml
    # Edit config/config.yaml with your settings
    cp examples/example_constants.yaml config/constants.yaml
    # Edit config/config.yaml with your settings


New Stars
---------
For some scripts, the code requires a database of new star particles at each snapshot. 
This will need to be generated to use the SFR/ Muv functionality for those scripts. 
The location of the newstars files should be specified in ``config.yaml``. 
Coming soon: code to generate this file! 

FOF Algorithms
--------------
The scripts contained here are designed to post-process outputs from multiple FOF alorithm types in AREPO. 
In Williams et al (2025), we compare the results of baryon-focused FOF algorithms to dark-matter focused algorithms.
Currently, four versions are supported: 

* `DMP-GS` - Dark Matter Primary, Gas and Stars secondary (standard)
* `SGDMP` - Dark matter, gas, and stars primary
* `SP` - Star Primary 
* `SGP` - Stars and gas primary. 

Setup 
-----
For the code to work, the output files of AREPO's FOF algorithms must be provided in specified directories.
After the configuration files have been edited to reflect the current settings, run ``setup.py`` to generate the appropriate directories for the output files. 

.. code-block:: bash

    python setup.py

Within each FOF directory, empty directories will be created called ``bounded3`` and ``bounded3/indv_objs``. 
These directories will be used to store the output files of the boundedness/virialization scripts. 
The ``setup.py`` script should result in the following file structure inside the directory given under ``input_dir`` in ``config.yaml``:   

.. code-block:: bash

    .
    ├── DMP-GS-Sig0
    │   └── bounded3
    │       └── indv_objs
    ├── SGP-Sig0
    │   └── bounded3
    │       └── indv_objs
    |── additional FOF directories...

Once the directories are created, move the output files to the appropriate directories. 
These should be called ``snap-groupordered-<snap>.hdf5`` and ``fof-subhalo-tab_<snap>.hdf5``.
The file strucutre should look like this: 

.. code-block:: bash

    .
    ├── DMP-GS-Sig0
    │   ├── bounded3
    │   │   └── indv_objs
    │   ├── snap-groupordered-<snap>.hdf5
    │   └── fof-subhalo-tab-<snap>.hdf5
    ├── SGP-Sig0
    │   ├── bounded3
    │   │   └── indv_objs
    │   ├── snap-groupordered-<snap>.hdf5
    │   └── fof-subhalo-tab-<snap>.hdf5
    |── additional FOF directories...

Once this has run, the setup is complete. You can check to ensure the setup is correct by running `` python test_setup.py``. 



Run the Code
------------
You can now run the scripts in the `scripts` and `modules` directory!


Citation 
--------

.. image:: https://zenodo.org/badge/782623936.svg
  :target: https://doi.org/10.5281/zenodo.15701630

.. image:: https://img.shields.io/badge/arXiv-2502.17561-b31b1b.svg
  :target: https://arxiv.org/abs/2502.17561
