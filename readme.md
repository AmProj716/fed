# FedProx Federated Learning on MNIST

This repository provides a PyTorch-based implementation of the FedProx algorithm for federated learning (FL) using the MNIST dataset. The code simulates a federated learning system with 10 clients that are equi-distributed, meaning each client holds an equal portion of the dataset. Notably, this implementation does not use TensorFlow Federated.

## Overview

Federated Learning allows multiple clients to collaboratively train a machine learning model while keeping their data local. The FedProx algorithm, used in this implementation, introduces a proximal term in each client's loss function to address system heterogeneity and stabilize training.

**Key Features:**
- **Equi-distributed data partitioning:** The MNIST training dataset is split equally among 10 clients.
- **Local training with FedProx:** Each client optimizes a local copy of the model by incorporating a proximal term to penalize divergence from the global model.
- **Model averaging:** The global model is updated by averaging local models from all clients.
- **CNN Architecture:** A simple yet effective Convolutional Neural Network (CNN) is implemented for MNIST classification.

## Requirements

The following dependencies are required to run the code:

- **Python 3.6+**
- **PyTorch**
- **torchvision**
- **NumPy**

## Installation

You can install the necessary Python packages using pip. Run the following command:

```bash
pip install torch torchvision numpy
