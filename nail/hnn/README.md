This is the Hamiltonian Neural Network (HNN) code I've written for NAIL.  The HNN is a multi-layer perceptron (MLP) with an inductive bias.  The bias is the symplectic gradient of the network outputs, which is used to calculate loss instead of the actual outputs themselves.  It allows the network to learn both ordered and chaotic trajectories of conservative dynamical systems.

The module data.py is a fork of a fork of code from the Greydanus team at Google.  It generates the simulated training data, and forecasts orbits using either an integrator or the trained model.
