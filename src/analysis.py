import os 

# changing the working directory to the current file directory
script_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(script_dir)

# importing libraries
import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns, time, keras_lmu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide tf warnings

# import my functions
from models import train_and_evaluate, sRNN, LSTM, GRU, LMU

# Setting an specific seed so that the results are reproducible
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)

# ------------------------------------------------------------------------------------
# Data Preprocessing 
# ------------------------------------------------------------------------------------
(X_train_image, y_train_image), (X_test_image, y_test) = tf.keras.datasets.mnist.load_data() # import MNIST dataset. 60,000 training images and 10,000 testing images

# normalizing (pixel value range from 0 -> 255 to 0 -> 1) to enhance the model performance. Float16 is used to save memory.
X_train_normalized = (X_train_image / 255).astype(np.float16) # shape (60000, 28, 28)
X_test_normalized = (X_test_image / 255).astype(np.float16) # shape (10000, 28, 28)

# reshape the data: a vector of pixels
X_train_vec = X_train_normalized.reshape((X_train_normalized.shape[0], -1, 1)) # shape (60000, 784, 1)
X_test_vec = X_test_normalized.reshape((X_test_normalized.shape[0], -1, 1)) # shape (10000, 784, 1)

# permutation of the vector components
perm = rng.permutation(X_train_vec.shape[1]) # generates a random permutation of the indices
X_train_perm = X_train_vec[:, perm]
X_test = X_test_vec[:, perm]

# split training and validation
X_train = X_train_perm[0:50000]; X_valid = X_train_perm[50000:] 
y_train = y_train_image[0:50000]; y_valid = y_train_image[50000:]

# ------------------------------------------------------------------------------------
# General Parameters
# ------------------------------------------------------------------------------------
NEURONS = 256
DROPOUT = 0.3
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 1000
VERBOSE_MODELS = False
VERBOSE_TRAINING = True

# ------------------------------------------------------------------------------------
# Modelling
# ------------------------------------------------------------------------------------

# sRNN model
sRNN_model = sRNN(
    X_train, 
    sRNN_neurons = NEURONS, 
    dropout = DROPOUT, 
    verbose = VERBOSE_MODELS
    )

# LSTM model 
LSTM_model = LSTM(
    X_train,
    LSTM_neurons = NEURONS,
    dropout = DROPOUT,
    verbose = VERBOSE_MODELS
    )

# GRU model
GRU_model = GRU(
    X_train,
    GRU_neurons = NEURONS,
    dropout = DROPOUT,
    verbose = VERBOSE_MODELS
    )


# LMU models

# LMU model with sRNN cells
LMU_sRNN_model = LMU(
    X_train,
    hidden_cell_type = "sRNN",
    LMU_neurons = NEURONS,
    dropout = DROPOUT,
    verbose = VERBOSE_MODELS
    )

# LMU model with LSTM cells
LMU_LSMT_model = LMU(
    X_train,
    hidden_cell_type = "LSTM",
    LMU_neurons = NEURONS,
    dropout = DROPOUT,
    verbose = VERBOSE_MODELS
    )

# LMU model with GRU cells
LMU_GRU_model = LMU(
    X_train,
    hidden_cell_type = "GRU",
    LMU_neurons = NEURONS,
    dropout = DROPOUT,
    verbose = VERBOSE_MODELS
    )

# dictionary with all the models
models = {'sRNN': sRNN_model, 'LSTM': LSTM_model, 'GRU': GRU_model, 'LMU_sRNN': LMU_sRNN_model, 'LMU_LSTM': LMU_LSMT_model, 'LMU_GRU': LMU_GRU_model}

# ------------------------------------------------------------------------------------
# Training and Evaluation
# ------------------------------------------------------------------------------------

# Uncomment the models you want to train
models_train = {
    'sRNN',
    'LSTM',
    'GRU',
    'LMU_sRNN',
    'LMU_LSTM',
    'LMU_GRU'
    }

for model_name, model in models.items():
    if model_name not in models_train: continue
    
    print (f"Training {model_name} model")
    train_and_evaluate(
        model, 
        X_train, 
        y_train, 
        X_valid, 
        y_valid, 
        X_test, 
        y_test, 
        lr = LR, 
        epochs = EPOCHS, 
        batch_size = BATCH_SIZE, 
        verbose = VERBOSE_TRAINING
        )
