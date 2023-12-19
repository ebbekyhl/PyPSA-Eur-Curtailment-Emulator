
# PyPSA-Eur Curtailment Emulator 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository introduces a linear representation of curtailment outputs in PyPSA-Eur. The representation is particularly relevant for models lacking subannual resolution to represent renewable curtailment. The parameterization is currently limited to the following aspects:
- Wind and solar mix
- Deployment of short-duration battery storage
- Deployment of long-duration energy storage 

For the PyPSA-Eur framework, please refer to the [master branch](https://github.com/PyPSA/pypsa-eur). However, for clarity, relevant scripts adjusted for this project are included in this repository. These adjustments comprise:
- Equality constraints on renewable resources (`solve_networks.py`). Find the function `add_renewable_potential_target`.
- LDES technology, which has the properties of an H2 storage with electrolyzers and fuel cells but is only connected to the electricity bus to function as a pure electricity storage (`prepare_sector_network.py`)
- LDES technology cost assumptions (`costs_2030.csv`)
- Removal of existing capacities of hydropower to make it a greenfield capacity optimization (except transmission lines) with the intention to enhance generalizability (`prepare_sector_network.py`).

## Dependencies

To run this framework, the following modules need to be installed:
- pandas
- numpy
- openpyxl
- pycel

## How to Run the Full Framework 

1. Fetch PyPSA scenarios (Zenodo) or calculate from scratch using the modifications documented in the folder `PyPSA-Eur/`.

2. Run `scripts/metrics_from_pypsa.py`:
   - Calculate metrics such as renewable curtailment, backup capacity, system cost, etc. This script requires access to PyPSA-Eur network files.

Note that steps 1 and 2 have already been performed, with resulting files located in the subfolder `calculated_metrics/`. However, if additional parameters or metrics are desired, it's possible with access to the PyPSA-Eur network files. This allows the framework to run without necessarily requiring a new PyPSA-Eur execution.

3. Run `PyPSA_emulator.ipynb`:
   - This script creates an emulator based on metrics calculated in step 2. It then evaluates curtailment for a user-specified range of wind and solar penetration combined with a given level of short-duration (Li-ion battery) and long-duration (H2) energy storage deployment.

4. `MESSAGE_implementation.py`:
   - As a use-case of the curtailment emulator, we apply the tool to represent curtailment in the Integrated Assessment Model MESSAGEix-GLOBIOM based on scenarios obtained in PyPSA-Eur. This with the aim to capture the synergies between wind and solar PV with technological deployment, as well as the impact of electrifying other energy-consuming sectors. 
