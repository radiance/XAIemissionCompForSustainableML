# XAI for Sustainable ML

This repository contributes source code accompanying the manuscript entitled "XAI for Sustainable ML"


## Summary

This project shows the usage of CodeCarbon framework to compute energy consumed by three python models (classification, regresseion and image recognition) to compare the use of SHAP vs feature reduction. Methods developed for an ongoing scientific work entitled "XAI for Sustainable ML"


## Getting started

Python package requirements:

Used Python Version: 3.8

- codecarbon
- statsmodels
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- scipy
- shap
- opencv-python
- torchvision==0.10.1
- torch==1.9.1
- pyyaml
- ipython

Command to install dependencies using PIP:
    pip install codecarbon statsmodels pandas numpy matplotlib seaborn sklearn scipy shap opencv-python ipython pyyaml torchvision==0.10.1 torch==1.9.1

Alternatively, use Conda:

	conda config --add channels pytorch
	conda config --add channels conda-forge

    conda install codecarbon statsmodels pandas numpy matplotlib seaborn scikit-learn scipy shap opencv pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cpuonly pyyaml ipython -c pytorch -c conda-forge --no-channel-priority

If a different pytorch version or CUDA support is required, refer to their doc:
https://pytorch.org/get-started/previous-versions/#v191

Based on your environment it might also be possible to just create the environment based on our configuration file:
    
    conda env create -f conda_environment.yml

## Run codecarbon for all models:

execute "python runner.py" in the root directory of this repo. This will run all the models 

## Generate Plots only:

- place previously generated "emissions.csv" in the "results" subfolder
- execute "python runner.py" in that directory
- all plots will be generated and saved in the "results" directory
