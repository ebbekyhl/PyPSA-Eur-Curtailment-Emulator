
# PyPSA-Eur Curtailment Emulator 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository introduces a linear parameterization of the curtailment outputs in PyPSA-Eur. This work establishes an emulator that can be used by models that do not have subannual resolution to represent renewable curtailment. The parameterization is currently limited to the following aspects:
- wind and solar mix
- deployment of short-duration battery storage
- deployment of long-duration energy storage 

For the PyPSA-Eur framework, we refer to the [master branch](https://github.com/PyPSA/pypsa-eur). But, for clarity, we have included relevant scripts in this repository that contains the adjustments made for this project. These adjustments include:
- The equality constraints on the renewable resources (`solve_networks.py`).
- LDES technology which has the properties of a H2 storage with electrolyzers and fuel cells but only connected to the electricity bus to make it a pure electricity storage (`prepare_sector_network.py`).
- LDES technology cost assumptions (`costs_2030.cvs`).
- We remove the existing capacities of hydropower such that it is a greenfield capacity optimization (except transmission lines) with the intention to make it more generalizable (`prepare_sector_network.py`).

# Dependencies

To run this framework, the following modules need to be installed:
- pandas
- numpy
- openpyxl
- pycel

# How to run the full framework 

1. Fetch PyPSA scenarios (Zenodo) or calculate from scratch using the modifications documented in the folder `PyPSA-Eur/`"

2. `scripts/metrics_from_pypsa.py`
    - Run this script to calculate metrics such as renewable curtailment, backup capacity, system cost, etc. Requires access to PyPSA-Eur network files.

Note that step 1 and 2 are already performed, with the resulting files located in the subfolder "results/". However, if additional parameters or metrics are desired, this is possible with the access to the PyPSA-Eur network files. In that way, this framework can be run without necessarily requiring a new run of PyPSA-Eur.

3. `pypsa_emulator.ipynb`
    - This script creates the emulator based on the metrics calculated in step 2. It then evaluates the curtailment for a user-specified range of wind and solar penetration combined with a given level of short-duration (Li-ion battery) and long-duration (H2) energy storage deployment. 

To be added:

4. `implement_pypsa_emulator.py`
    - As a use-case of this tool, we implement the emulator in the Integrated Assessment Model MESSAGEix-GLOBIOM
