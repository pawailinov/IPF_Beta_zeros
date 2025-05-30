# Keep it Simple: Addressing Rare Events in Data Synthesis Using Beta Divergence

This repository contains the implementation and experimental analysis of an extended Iterative Proportional Fitting (IPF) algorithm that incorporates β-divergence optimization. The project investigates the advantages of using β-divergence in generating synthetic population in the case of rare events. 

## Repository Structure

- `Beta_optimization.py`: Core implementation of the β-divergence optimization algorithm
- `KL_optimization.py`: Implementation of traditional KL-divergence based IPF
- `toy_example_several_betas.ipynb`: Analysis with synthetic data using different β values
- `toy_example_low_values.ipynb`: Analysis with synthetic data using low imputed values
- `real_data_several_betas.ipynb`: Analysis with MTMC data using different β values
- `real_data_low_values.ipynb`: Analysis with MTMC data using low imputed values
- `indiv_rw_2021.csv`: Swiss Mobility and Transport micro-census (MTMC) dataset from 2021

## Requirements

- NumPy
- Matplotlib (for visualization)
- Jupyter (for running notebooks)