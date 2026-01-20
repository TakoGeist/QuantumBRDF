# Parametrized Quantum circuits for estimating BRDFs

This repository contains the code developed for the master’s thesis **“Parametrized Quantum circuits for estimating BRDFs”**, which explores the feasibility of using **Parametrized Quantum Circuits (PQCs)** to model Bidirectional Reflectance Distribution Functions (BRDFs). The work investigates whether quantum machine learning techniques can approximate classical reflectance models, providing a proof of concept for future quantum-based approaches in physically based rendering.

---

## Overview

The goal of this project is to study how PQCs can be employed to learn angular reflectance behavior typically described by BRDFs. The work focuses on progressively more complex reflectance models:

- **Lambert’s model**, representing ideal diffuse reflection
- **Phong’s model**, combining diffuse and specular reflection with tunable shininess
- **Oren–Nayar’s model**, modeling rough diffuse surfaces with nonlinear and discontinuous behavior

These models are implemented and analyzed using hybrid quantum–classical workflows, through simulated quantum circuits.

---

## Repository Structure

The repository consists of notebooks and scripts used to generate data, define PQC architectures, train models, and analyze results.

- `lambert.ipynb`  
  Experiments modeling Lambert’s BRDF using simple quantum circuits.

- `phong_*.ipynb`  
  A collection of notebooks exploring different configurations and parameter regimes of Phong’s model, including data re-uploading and input scaling strategies.

- `oren_nayar.py`  
  Script implementing experiments for the Oren–Nayar BRDF.

- `oren_nayar_simplified.ipynb`  
  Simplified notebook experiments for analyzing Oren–Nayar behavior.

- `model.py`  
  Core implementation of PQC architectures.

- `generator.py`  
  Utilities for generating synthetic BRDF datasets and sampling angular inputs.

- `nb_creator.py`  
  Helper script used to create parameterized experiment notebooks.

- `packages.txt`  
  List of required Python dependencies.

Most notebooks include the full experimental pipeline: data generation, PQC definition, training, evaluation, and visualization.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/TakoGeist/QuantumBRDF.git
   cd QuantumBRDF


## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

### Final Notes

Code structure is a consequence of Jupyter notebooks inability to import modules from parent directories without altering `sys.path`.