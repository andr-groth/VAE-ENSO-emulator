# Efficient inference and learning of a generative model for ENSO predictions from large multi-model datasets

## Overview

Jupyter notebooks for the implementation of a variational autoencoder (VAE) for ENSO modeling and prediction.

The work is published in

> Groth A. and E. Chavez, 2024:  Efficient inference and learning of a generative model for ENSO predictions from large multi-model datasets. _Climate Dynamics_, accepted, https://doi.org/10.1007/s00382-024-07162-w.

In this paper, historical simulations of global sea-surface temperature (SST) from the fifth phase of the Coupled Model Intercomparison Project (CMIP5) are analyzed. Based on the concept of a variational auto-encoder (VAE), a generative model of global SST is proposed in combination with an inference model that aims to solve the problem of determining a joint distribution over the data generating factors. With a focus on the El Niño Southern Oscillation (ENSO), the performance of the VAE-based approach in simulating various central features of observed ENSO dynamics is demonstrated. A combination of the VAE with a forecasting model is proposed to make predictions about the distribution of global SST and the corresponding future path of the Niño index from the learned latent factors.

## Requirements

1. The Jupyter notebooks require the __VAE package__, which is available at:

    > https://github.com/andr-groth/VAE-project

2. Sample data used in [Groth and Chavez (2024)](https://doi.org/10.1007/s00382-024-07162-w) is included in the [`data/`](/data/) folder. The data was collected with the help of the Climate Explorer at:

    > http://climexp.knmi.nl

    For more information on the data see [`data/README.md`](/data/README.md).

## Examples

Example runs of the Jupyter notebooks are available in the [`examples/`](/examples/) folder of this repository. The examples are based on the sample data in the [`data/`](/data/) folder.