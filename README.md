# DASS #

This is the implementation repository of paper, Not All Synthetic Vulnerabilities Are Equal: Synthetic
Vulnerability Selection for Better Learning Deep Vulnerability
Detectors


## Description ##

We propose DASS, 

## Reproducibility ##
### Requirements ###
- Python==3.7.13
- torch==1.12.1
- transformers==4.25.1
- tqdm==4.62.3
- numpy==1.21.5
- scikit-learn==1.0.2
- sinularity-ce==4.2.2 (for running the evaluation code)

### Structure ###
    |-DASS/ "implementation for DASS".
    |-evaluation/ "contains the code for evaluating the DASS and baseline approaches."
    |   |-devign/ 
    |   |-linevul/
    |   |-linevd/
    |   |-velvet/
    |-results/ "tables of experimental results."

### Usage ###