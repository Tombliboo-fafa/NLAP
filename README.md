# Neighborhood-guided Label Augmentation and Pruning (NLAP)

Implementation of the paper:  
**"Neighbor-aware Label Refinement: Enhancing Unreliable Instance-Dependent Partial Labels"**

This repository provides the code for reproducing the experiments in the paper, including generating UIDPLL datasets, training, and evaluation.

Before running any experiments:

1. Place your dataset `data_name` under: ./data/benchmark/data_name/
2. Place the pretrained model used for UIDPLL data generation under: ./trained_model/clean/data_name.pt

## Running the Demo

A demo script `run.sh` is provided for testing the pipeline:

```bash
sh ./run.sh
```

This will run a simple example demonstrating data loading, NLAP refinement, and evaluation.
