# Python grid estimation

These scripts comprise statistical estimation algorithms to estimate the grid topology based on sensor measurements at nodes in the network.

Specifically, they call the Chow-Liu algorithm to estimate the power grid topology based on a binary tree structure assumption. The algorithm uses the voltage magnitude, and/or phase and computes joint probablities between nodes. It then computes the mutal information between nodes. This is done via a discretization of the probability, or via an estimator or by modeling the voltage distributions as mixed Gaussian models (GMMs). Using the mutual information, we attach nodes together while ensuring we create no cycles in the predicted graph structure.


## Installing the package
Please use pip install the library.

```pip install grid_top_est```

## Juptyer notebook

Please see the juptyer notebook (examples/Grid_Topology_Estimation.ipynb) to get a better how this library can be used in practice. In the notebook an 8 node IEEE example is shown as well as an example main file.



