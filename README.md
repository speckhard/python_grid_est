# Python grid estimation

These scripts comprise statistical estimation algorithms to estimate the grid topology based on sensor measurements at nodes in the network.

Specifically, they call the Chow-Liu algorithm to estimate the power grid topology based on a binary tree structure assumption. The algorithm uses the voltage magnitude, and/or phase and computes joint probablities between nodes. It then computes the mutal information between nodes. This is done via a discretization of the probability, or via an estimator or by modeling the voltage distributions as mixed Gaussian models (GMMs). Using the mutual information, we attach nodes together while ensuring we create no cycles in the predicted graph structure.


