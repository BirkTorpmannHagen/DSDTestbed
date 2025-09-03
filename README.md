# Test Bench for Distributional Shift Detector Research
Utilities for running and evaluating OoD detectors in the context of distributional shift detection. 

## How to use
This repository is organized in a manner intended to facilitate modularity. The following directories can be 
augmented with your own code depending on your desired use case.
 -  ```datasets/``` - Implementations of dataloaders
 - ```classifier/```, ```segmentor/```, etc. code that defines neural network architectures and training routines with 
   PyTorch lightning. 
 - ```testbeds/``` classes that define InD and OoD data loaders, loads checkpoints, and computes losses/metrics. 
 - ```experiments/``` defines code for plotting and collecting metrics

We also implement the following:
 - ```bias_samplers.py``` defines BatchSampler objects for assessing the performance of batched OoD detectors under bias
 - ```ood_detector.py``` defines how OoD detectors are calibrated. Assumes a pre-computed pandas dataframe of OoD 
   detector features.
 - ```ooo_detector_computation.py``` code for computing OoD detector features using testbeds.
 - ```ood_detector_features.py``` defines how OoD detector features are computed.
 - ```eval_detectors.py``` contain data collection scripts, resulting in .csv files/pandas dataframes of raw feature 
   values
 - ```utils.py``` various utils and constants
 - ```experiments.py``` runs experiments and plots from ```experiments/```

Questions can be directed to bto033@uit.no, or birk.torpmann.hagen@gmail.com