# RNN Analysis of the psMNIST Dataset

This repository contains the implementation and analysis of various Recurrent Neural Network (RNN) architectures, including simple RNN (sRNN), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Legendre Memory Unit (LMU) hybrids, on the permuted sequential MNIST (psMNIST) dataset. The study aims to evaluate the performance, interpretability, and computational efficiency of these architectures, providing insights into their capabilities and limitations when handling complex sequential data. 

## Repository Contents

In the `src/` directory, you will find the following files: 

### `models.py`

This file contains the definitions of the RNN models and training used in the study.
- `train_and_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, lr=0.001, epochs=100, batch_size=1000, verbose=False)`: Trains and evaluates a given model on the provided training, validation, and test datasets.
- `sRNN(X_train, sRNN_neurons=256, dropout=0.3, rec_dropout=0.0, verbose=False)`: Defines a Simple RNN model.
- `LSTM(X_train, LSTM_neurons=256, dropout=0.3, rec_dropout=0.0, verbose=False)`: Defines a Long Short-Term Memory model.
- `GRU(X_train, GRU_neurons=256, dropout=0.3, rec_dropout=0.0, verbose=False)`: Defines a Gated Recurrent Unit model.
- `LMU(X_train, hidden_cell_type="sRNN", LMU_neurons=256, dropout=0.3, rec_dropout=0.0, verbose=False)`: Defines a Legendre Memory Unit model with a specified hidden cell type.

### `analysis.py` & `train.py`
These files include functions for training and evaluating the defined RNN models. `train.py` in particular is thought to be used from the terminal. 

### `analysis.ipynb`

A Jupyter notebook that contains the analysis of the training results, including performance metrics, confusion matrices, and weight distributions. This notebook provides a detailed breakdown of the findings and visualizations to support the interpretability of the models.

