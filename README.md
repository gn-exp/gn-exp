# Explainability Techniques for Graph Convolutional Networks

Anonymous repository accompanying the paper "Explainability Techniques for Graph Convolutional Networks" submitted to the ICML 2019 Workshop ["Learning and Reasoning with Graph-Structured Data"](https://graphreason.github.io/).

## Structure
- `src`, `config`, `data` contain code, configuration files and data for the experiments
  - `infection`, `solubility` contain the code for the two experiments in the paper
  - `torchgraphs` contain the core graph network library
  - `guidedbackrprop`, `relevance` contain the code to run Guided Backpropagation and Layer-wise Relevance Propagation on top of PyTorch's `autograd`
- `notebooks`, `models` contain a visualization of the datasets and the results of the experiments
- `test` contains unit tests for the `torchgraphs` module (core GN library)
- `conda.yaml` contains the conda environment for the project

## Setup
The project is build on top of Python 3.7, PyTorch 1+ and many other open source projects.

```bash
conda env create -f conda.yaml
conda activate gn-exp
python setup.py develop
pytest
```

## Training

- Infection: see this [readme](./src/infection/notes.md)
- Solubility: see this [readme](./src/solubility/notes.md)

## Experimental results

See the notebooks in [`notebooks`](./notebooks)
```bash
conda activate gn-exp
cd notebooks
jupyter lab 
```

## Testing
 
Unit tests for the Graph Network library (`torchgraphs` module):
```bash
conda env create -f conda.yaml
conda activate gn-exp
python setup.py develop
pytest
```