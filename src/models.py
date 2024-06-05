import os 

# changing the working directory to the current file directory
script_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(script_dir) 

# importing libraries
import pandas as pd, tensorflow as tf, time, keras_lmu

# ------------------------------------------------------------------------------------
# Training and Evaluation
# ------------------------------------------------------------------------------------
def train_and_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, lr =0.001, epochs = 100, batch_size = 1000, verbose = False):
    """
    Trains and evaluates a given model on provided training, validation, and test data.

    Parameters:
    - model (tf.keras.Model): The TensorFlow/Keras model to be trained and evaluated.
    - X_train (np.ndarray): Training feature data.
    - y_train (np.ndarray): Training labels.
    - X_valid (np.ndarray): Validation feature data.
    - y_valid (np.ndarray): Validation labels.
    - X_test (np.ndarray): Test feature data.
    - y_test (np.ndarray): Test labels.
    - lr (float): Learning rate for the optimizer (default: 0.001).
    - epochs (int): Number of epochs for training (default: 100).
    - batch_size (int): Batch size for training (default: 1000).
    - verbose (bool): If True, prints detailed information during training (default: False).

    Returns:
    None
    """
    
    if verbose: 
        # Data shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Hyperparameters
        print(f"Learning Rate: {lr}, Epochs: {epochs}, Batch size: {batch_size}")
    
    model.name = model.name; 
    hyperparameters = f'lr_{lr}_epochs_{epochs}_batch_{batch_size}'
    job_specifics = f"{model.name}_{hyperparameters}"
    
    # TensorFlow model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        # weighted_metrics=None,
        # run_eagerly=False,
        # steps_per_execution=1,
        # jit_compile='auto',
        # auto_scale_loss=True
    )
    
    # print the model summary
    model.summary()
    
    # save checkpoints
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./Results/weights_{job_specifics}.hdf5", # .keras
            monitor="val_loss",
            verbose=1,
            save_best_only=True
                ),
        ]
    
    t0 = time.time()

    # training the model
    result = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            workers=16,
            use_multiprocessing=True,
            verbose=2,
        )

    training_time = time.time() - t0
    
    if verbose: print(f"Training time:, {training_time/60} minutes")

    # save the results
    dataframe = pd.DataFrame.from_dict(result.history, orient = 'index')
    dataframe.to_csv(f"./Results/results_{job_specifics}.txt", header=False)

    # save model
    model.save(f"./Results/model_{job_specifics}")

# ------------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------------

# Simple RNN model (sRNN)
def sRNN(X_train, sRNN_neurons = 256, dropout = 0.3, rec_dropout = 0.0, verbose = False):
    """
    Defines a Simple RNN (sRNN) model.

    Parameters:
    - X_train (np.ndarray): Training feature data to determine the input shape.
    - sRNN_neurons (int): Number of neurons in the Simple RNN layer (default: 256).
    - dropout (float): Dropout rate for the Simple RNN layer (default: 0.3).
    - rec_dropout (float): Recurrent dropout rate for the Simple RNN layer (default: 0.0).
    - verbose (bool): If True, prints detailed information about the model (default: False).

    Returns:
    - model_sRNN (tf.keras.Model): A TensorFlow/Keras Sequential model with a Simple RNN layer.
    """
    
    if verbose: print(f"Number of sRNN neurons: {sRNN_neurons}, dropout: {dropout}")
    
    n_pixels = X_train.shape[1]
    
    # Define layers and model for sRNN, default values are commented.
    sRNNlayer = tf.keras.layers.SimpleRNN(
        units = sRNN_neurons,
        # activation='tanh',
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # recurrent_initializer='orthogonal',
        # bias_initializer='zeros',
        # kernel_regularizer=None,
        # recurrent_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # recurrent_constraint=None,
        # bias_constraint=None,
        dropout = dropout,
        recurrent_dropout = rec_dropout,
        # return_sequences=False,
        # return_state=False,
        # go_backwards=False,
        # stateful=False,
        # unroll=False,
        # seed=None,
        input_shape = (n_pixels, 1)
    )
    
    final_layer = tf.keras.layers.Dense(
        units = 10,
        # activation = None, # the most efficient is to consider the softmax, but we do specify that in the compiling step
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # bias_initializer='zeros',
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # lora_rank=None,
    )
    
    # Defining the sequential model
    model_sRNN = tf.keras.models.Sequential(
        [
            sRNNlayer,
            tf.keras.layers.BatchNormalization(), 
            final_layer
        ],
        name="SimpleRNN"
    )
    return model_sRNN

# LSTM model
def LSTM(X_train, LSTM_neurons = 256, dropout = 0.3, rec_dropout = 0.0, verbose = False):
    """
    Defines a Long Short-Term Memory (LSTM) model.

    Parameters:
    - X_train (np.ndarray): Training feature data to determine the input shape.
    - LSTM_neurons (int): Number of neurons in the LSTM layer (default: 256).
    - dropout (float): Dropout rate for the LSTM layer (default: 0.3).
    - rec_dropout (float): Recurrent dropout rate for the LSTM layer (default: 0.0).
    - verbose (bool): If True, prints detailed information about the model (default: False).

    Returns:
    - model_LSTM (tf.keras.Model): A TensorFlow/Keras Sequential model with an LSTM layer.
    """
    
    if verbose: print(f"Number of LSTM neurons: {LSTM_neurons}, dropout: {dropout}")
    
    n_pixels = X_train.shape[1]
    
    # Define layers and model for LSTM, default values are commented.
    LSTM_layer = tf.keras.layers.LSTM(
        units = LSTM_neurons,
        # activation='tanh',
        # recurrent_activation='sigmoid',
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # recurrent_initializer='orthogonal',
        # bias_initializer='zeros',
        # unit_forget_bias=True,
        # kernel_regularizer=None,
        # recurrent_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # recurrent_constraint=None,
        # bias_constraint=None,
        dropout = dropout,
        recurrent_dropout = rec_dropout,
        # return_sequences=False,
        # return_state=False,
        # go_backwards=False,
        # stateful=False,
        # unroll=False,
        # time_major=False,
        # reset_after=False,
        # **kwargs,
        input_shape = (n_pixels, 1)
    )
    
    final_layer = tf.keras.layers.Dense(
        units = 10,
        # activation = None, # the most efficient is to consider the softmax, but we do specify that in the compiling step
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # bias_initializer='zeros',
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # lora_rank=None,
    )
    
    # Defining the sequential model
    model_LSTM = tf.keras.models.Sequential(
        [
            LSTM_layer,
            tf.keras.layers.BatchNormalization(), 
            final_layer
        ],
        name="LSTM"
    )
    return model_LSTM

def GRU(X_train, GRU_neurons = 256, dropout = 0.3, rec_dropout = 0.0, verbose = False):
    """
    Defines a Gated Recurrent Unit (GRU) model.

    Parameters:
    - X_train (np.ndarray): Training feature data to determine the input shape.
    - GRU_neurons (int): Number of neurons in the GRU layer (default: 256).
    - dropout (float): Dropout rate for the GRU layer (default: 0.3).
    - rec_dropout (float): Recurrent dropout rate for the GRU layer (default: 0.0).
    - verbose (bool): If True, prints detailed information about the model (default: False).

    Returns:
    - model_GRU (tf.keras.Model): A TensorFlow/Keras Sequential model with a GRU layer.
    """
    
    if verbose: print(f"Number of GRU neurons: {GRU_neurons}, dropout: {dropout}")
    
    n_pixels = X_train.shape[1]
    
    # Define layers and model for GRU, default values are commented.
    GRU_layer = tf.keras.layers.GRU(
        units = GRU_neurons,
        # activation='tanh',
        # recurrent_activation='sigmoid',
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # recurrent_initializer='orthogonal',
        # bias_initializer='zeros',
        # unit_forget_bias=True,
        # kernel_regularizer=None,
        # recurrent_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # recurrent_constraint=None,
        # bias_constraint=None,
        dropout=dropout,
        recurrent_dropout= rec_dropout,
        # return_sequences=False,
        # return_state=False,
        # go_backwards=False,
        # stateful=False,
        # unroll=False,
        # time_major=False,
        # reset_after=False,
        input_shape=(n_pixels, 1)
    )
    
    final_layer = tf.keras.layers.Dense(
        units = 10,
        # activation = None, # the most efficient is to consider the softmax, but we do specify that in the compiling step
        # use_bias=True,
        # kernel_initializer='glorot_uniform',
        # bias_initializer='zeros',
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # lora_rank=None,
    )
    
    # Defining the sequential model
    model_GRU = tf.keras.models.Sequential(
        [
            GRU_layer,
            tf.keras.layers.BatchNormalization(), 
            final_layer
        ],
        name="GRU"
    )
    return model_GRU
    
    
def LMU(X_train, hidden_cell_type = "sRNN", LMU_neurons = 256, dropout = 0.3, rec_dropout = 0.0, verbose = False):
    """
    Defines a Legendre Memory Unit (LMU) model with a specified hidden cell type.

    Parameters:
    - X_train (np.ndarray): Training feature data to determine the input shape.
    - hidden_cell_type (str): Type of hidden cell to use ('sRNN', 'LSTM', or 'GRU') (default: 'sRNN').
    - LMU_neurons (int): Number of neurons in the LMU hidden cell (default: 256).
    - dropout (float): Dropout rate for the LMU hidden cell (default: 0.3).
    - rec_dropout (float): Recurrent dropout rate for the LMU hidden cell (default: 0.0).
    - verbose (bool): If True, prints detailed information about the model (default: False).

    Returns:
    - model_LMU (tf.keras.Model): A TensorFlow/Keras Model with an LMU layer.
    """
    
    if verbose: print(f"Number of LMU neurons: {LMU_neurons}, dropout: {dropout}")
    
    if hidden_cell_type == "sRNN": hidden_cell = tf.keras.layers.SimpleRNNCell(units = LMU_neurons)
    
    elif hidden_cell_type == "LSTM": hidden_cell = tf.keras.layers.LSTMCell(units = LMU_neurons)
    
    elif hidden_cell_type == "GRU": hidden_cell = tf.keras.layers.GRUCell(units = LMU_neurons)
    
    else: raise ValueError("hidden_cell_type must be 'sRNN', 'LSTM' or 'GRU'")
    
    n_pixels = X_train.shape[1]
    
    inputs = tf.keras.Input((n_pixels, 1))
    
    LMU_layer = tf.keras.layers.RNN(
        keras_lmu.LMUCell(
            memory_d = 1,
            order = LMU_neurons,
            theta = n_pixels,
            hidden_cell = hidden_cell,
            hidden_to_memory=False,
            memory_to_memory=False,
            input_to_hidden=True,
            kernel_initializer="ones",
            dropout = dropout,
            recurrent_dropout = rec_dropout
            )
        )
    complete_layer = LMU_layer(inputs)
    
    outputs = tf.keras.layers.Dense(10)(complete_layer)
    
    # Defining the model
    model_LMU = tf.keras.Model(
        inputs=inputs, 
        outputs=outputs,
        name = f"LMU_{hidden_cell_type}"
        )
    
    return model_LMU 