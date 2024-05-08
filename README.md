# Efficient inference and learning of a generative model for ENSO predictions from large multi-model datasets

<p align="center">
  <img src="/examples/VAEp_explore_files/VAEp_explore_109_0.png" height="300"/>
  <img src="/examples/VAEp_explore_files/VAEp_explore_185_0.png" height="300"/>
</p>


## Overview

Official Jupyter notebooks for the implementation of a variational autoencoder (VAE) for ENSO modeling and prediction.

The work is published in

> Groth and Chavez, 2024:  Efficient inference and learning of a generative model for ENSO predictions from large multi-model datasets. _Climate Dynamics_, <https://doi.org/10.1007/s00382-024-07162-w>.

In this paper, historical simulations of global sea-surface temperature (SST) from the fifth phase of the Coupled Model Intercomparison Project (CMIP5) are analyzed. Based on the concept of a variational auto-encoder (VAE), a generative model of global SST is proposed in combination with an inference model that aims to solve the problem of determining a joint distribution over the data generating factors. With a focus on the El Niño Southern Oscillation (ENSO), the performance of the VAE-based approach in simulating various central features of observed ENSO dynamics is demonstrated. A combination of the VAE with a forecasting model is proposed to make predictions about the distribution of global SST and the corresponding future path of the Niño index from the learned latent factors.

## Requirements

1. The Jupyter notebooks require the __VAE package__, which is available at:

    > <https://github.com/andr-groth/VAE-project>

2. Sample data used in [Groth and Chavez (2024)](https://doi.org/10.1007/s00382-024-07162-w) is included in the [`data/`](/data/) folder. The data was collected with the help of the Climate Explorer at:

    > <http://climexp.knmi.nl>

    For more information on the data see [`data/README.md`](/data/README.md).

## Usage

The repository includes two Jupyter notebooks, one for model training and another one for model exploration.

1. The model is trained with [`VAEp_train.ipynb`](/VAEp_train.ipynb)
2. Properties of the trained model are explored with [`VAEp_explore.ipynb`](/VAEp_explore.ipynb)

The weights of a trained model used to create the figures in [Groth and Chavez (2024)](https://doi.org/10.1007/s00382-024-07162-w) are provided in the [`logs/`](/logs/) folder of this repository.

Example runs of the Jupyter notebooks are available in the [`examples/`](/examples/) folder of this repository. The examples are based on the sample data in the [`data/`](/data/) folder.

## Reference
Please add a reference to the following paper if you use parts of this code:

```
@Article{Groth.Chavez.2024,
  author           = {Groth, Andreas and Chavez, Erik},
  journal          = {Climate Dynamics},
  title            = {Efficient inference and learning of a generative model for {ENSO} predictions from large multi-model datasets},
  year             = {2024},
  doi              = {10.1007/s00382-024-07162-w},
  publisher        = {Springer Science and Business Media LLC},
}
```
