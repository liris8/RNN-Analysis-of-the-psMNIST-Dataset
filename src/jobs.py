import os 

# changing the working directory to the current file directory
script_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(script_dir)

# importing libraries
import numpy as np, pandas as pd, tensorflow as tf, time, keras_lmu, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide tf warnings

# import my functions
from models import sRNN, LSTM, GRU, LMU

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
# Modelling 
# ------------------------------------------------------------------------------------

def main():
    # example to run with command-line arguments:
    # python submit.py --NN GRU --dropout 0.2 --rec_dropout 0.5 
    # python submit_test.py --NN GRU --epochs 1 --neurons 10 --lr 0.01
    print('entering the main function')

    # define the command line argunments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir', type=str, default='./Results/',
        # default='/Users/luisirisarri/Documents/Home/1 Projects/2nd Semester/Applied Data Analysis and Machine Learning/My Notes/Projects/2-Artificial Neural Networks/Coding/Results',
        help='output path')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='number of epochs')
    parser.add_argument(
        '--batch', type=int, default=1000,
        help='batch size')
    parser.add_argument(
        '--neurons', type=int, default=256,
        help='number of neurons')
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='dropout')
    parser.add_argument(
        '--rec_dropout', type=float, default=0.5,
        help='recurrent dropout')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '--NN', type=str, default='LMU',
        help='Architecture')

    # collect arguments
    args = parser.parse_args()

    OUTDIR      = args.outdir
    NEURONS     = args.neurons
    DROPOUT     = args.dropout
    REC_DROPOUT = args.rec_dropout
    EPOCHS      = args.epochs
    BATCH       = args.batch
    LR          = args.lr
    NN_TYPE     = args.NN

    folder_path = OUTDIR + NN_TYPE
    params = "NN_" + NN_TYPE + "_ep_" + str(EPOCHS) + "_ba_" + str(BATCH) + "_neu_" + str(NEURONS) + "_lr_" + str(LR) + "_do_" + str(DROPOUT) + "_rdo_" + str(REC_DROPOUT)

    # this lines create a directory (if it does not already exist) to store the outputs
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else: pass

    # Now we create the network taking into account the arguments given

    if NN_TYPE == 'sRNN':
        
        model = sRNN(
                    X_train, 
                    sRNN_neurons = NEURONS, 
                    dropout = DROPOUT, 
                    rec_dropout=REC_DROPOUT,
                    verbose = True
                    )

    if NN_TYPE == 'LSTM':
        
       model = LSTM(
                    X_train,
                    LSTM_neurons = NEURONS,
                    dropout = DROPOUT,
                    rec_dropout=REC_DROPOUT,
                    verbose = True
                    )
        

    if NN_TYPE == 'GRU':
        
        model = GRU(
                    X_train,
                    GRU_neurons = NEURONS,
                    dropout = DROPOUT,
                    rec_dropout=REC_DROPOUT,
                    verbose = True
                    )
    
    if NN_TYPE[:3] == 'LMU':
        
        model = LMU(
                    X_train,
                    hidden_cell_type = NN_TYPE[4:],
                    LMU_neurons = NEURONS,
                    dropout = DROPOUT,
                    verbose = True
                    )
       

    print('architecture selected')

    # TensorFlow model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
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
            filepath=f"{folder_path}/weights_{params}.hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True
            # save_weights_only=False,
            # mode='auto',
            # save_freq='epoch',
            # initial_value_threshold=None
            ),
        ]
    
    t0 = time.time()

    # Training the model
    result = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            workers=16,
            use_multiprocessing=True,
            verbose=2,
        )

    training_time = time.time() - t0
    
    print(f"Training time:, {training_time}")

    # save the results
    dataframe = pd.DataFrame.from_dict(result.history, orient = 'index')
    dataframe.to_csv(f"{folder_path}/results_{params}.txt", header=False)

    # save model
    model.save(f"{folder_path}/model_{params}")