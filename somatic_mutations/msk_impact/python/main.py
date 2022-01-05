
import os
import argparse
import itertools
import random
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, LocallyConnected1D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # use a non-interactive backend
# matplotlib.use('macosx')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def plot_loss(m1, start, plot_path):
    # plot the training loss and accuracy
    end = len(m1.history['loss'])
    N   = np.arange(start, end)
    s   = slice(start,end)
    
    plt.style.use("ggplot")
    plt.figure(figsize=(4, 3), dpi=300)
    
    plt.plot(N, (m1.history["loss"][s]), label="train_loss")
    plt.plot(N, (m1.history["val_loss"][s]), label="val_loss")
    
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.2)
    plt.savefig(plot_path)
    plt.close()


# input_file   = "mut_matrix_340_x_9112.txt"
# data_dir     = "../data/"
# fig_dir      = "../figures"
# results_dir  = "../results"
# latent_size  = 20
# hidden_size1 = 200
# hidden_size2 = 100
# n_epochs     = 50
# learn_rate   = 1e-3
# batch_size   = 32
# testing_frac = 0.1
# n_layers     = 1

def main(input_file, data_dir = "data", fig_dir = "figures", 
    results_dir = "results", split_data_set = True, testing_frac = 0.2, 
    latent_size = 20, hidden_size1 = 400, hidden_size2 = 200, 
    n_layers=1,  n_epochs = 50, batch_size = 32, learn_rate = 1e-3):

    if n_layers > 2 :
            print('n_layers can be only 1 or 2:')
            raise ValueError

    hidden_sizes = [hidden_size1, hidden_size2]

    # -----------------------------------------------------------------
    # model configuration
    # -----------------------------------------------------------------

    config = str(latent_size) + '_' + str(hidden_sizes[0])

    if n_layers == 2:
             config = config + '_' + str(hidden_sizes[1])

    config = config + '_bs_' + str(batch_size) + '_lr_' + str(learn_rate)
    config = config + '_e_' + str(n_epochs) + '_layer_' + str(n_layers)

    # -----------------------------------------------------------------
    # read in data
    # -----------------------------------------------------------------

    file_name = os.path.join(data_dir, input_file)

    print('input file:')
    print(file_name)
    print('')

    df = pd.read_csv(file_name, sep='\t', header=0, index_col=0)
    df.iloc[0:3,0:5]

    # -----------------------------------------------------------------
    # filter samples by mutation load
    # -----------------------------------------------------------------

    mload = df.sum(0)
    mload.shape
    print('mutation load:')
    print(mload.describe())

    # how many samples have 5 or more mutated genes
    mload.ge(5).sum()

    df = df.loc[:,mload.ge(5)]
    df.shape

    mload = np.array(df.sum(0))

    n, bins, patches = plt.hist(np.log10(mload), 50, density=False, facecolor='g', alpha=0.75)

    hist_path = os.path.join(fig_dir, '_mutation_load_hist.png')

    plt.xlabel('log10(mutation laod)')
    plt.ylabel('frequnecy')
    plt.grid(True)
    plt.savefig(hist_path)

    # -----------------------------------------------------------------
    # split_training_testing_data
    # -----------------------------------------------------------------

    dfT = df.T
    trainX, testX = train_test_split(dfT, test_size=testing_frac, random_state=1999)

    print('training and testing data dimension:')
    print(trainX.shape)
    print(testX.shape)

    print('trainX[0:2, 0:9]:')
    print(trainX.iloc[0:2, 0:9])

    print('testX[0:2, 0:9]:')
    print(testX.iloc[0:2, 0:9])


    # -----------------------------------------------------------------
    # model setup
    # -----------------------------------------------------------------

    input_e = Input(shape=(trainX.shape[1],), name='input')
    encoded = Dense(hidden_sizes[0], activation='relu', name='encoding1')(input_e)
    encoded = Dropout(rate=0.5, name='dropout_encoding')(encoded)

    if n_layers > 1: 
            encoded = Dense(hidden_sizes[1], activation='relu', name='encoding2')(encoded)

    encoded = Dense(latent_size, activation='relu', name='bottleneck')(encoded)
    decoded = Dense(hidden_sizes[0], activation='relu', name='decoding1')(encoded)

    if n_layers > 1: 
            decoded = Dense(hidden_sizes[1], activation='relu', name='decoding2')(decoded)

    decoded = Dense(trainX.shape[1], activation='sigmoid', name='output')(decoded)

    autoencoder = Model(input_e, decoded)
    autoencoder.summary()

    # -----------------------------------------------------------------
    # model fitting
    # -----------------------------------------------------------------

    adam1 = optimizers.Adam(lr=learn_rate, beta_1=0.8, beta_2=0.9)
    adam1 = optimizers.Adam(lr=learn_rate)
    # sgd1  = optimizers.SGD(lr=learn_rate, momentum=0.8, decay=0.0, nesterov=False)

    autoencoder.compile(optimizer=adam1, loss='binary_crossentropy', metrics=['accuracy'])

    m1 = autoencoder.fit(trainX, trainX, epochs=n_epochs, batch_size=batch_size, 
        verbose=2, validation_data=(testX, testX))

    train_pred = autoencoder.predict(trainX)
    test_pred  = autoencoder.predict(testX)

    print('')
    print('training and testing classificiation accuracy:')
    print('')

    print('training:')
    print(pd.crosstab(train_pred.flatten() > 0.5, np.array(trainX).flatten()))
    print('')

    print('testing:')
    print(pd.crosstab(test_pred.flatten() > 0.5, np.array(testX).flatten()))
    print('')

    plot_path  = os.path.join(fig_dir, 'track_loss_' + config + '.png')
    plot_loss(m1, 1, plot_path)

    # model_path = os.path.join(results_dir, 'model_' + config + '.h5')
    # autoencoder.save(model_path)

    # -----------------------------------------------------------------
    # extract final model output
    # -----------------------------------------------------------------

    out_train = pd.DataFrame(data=train_pred, index=trainX.index)
    out_test  = pd.DataFrame(data=test_pred,  index=testX.index)


    fnm = 'model_' + config + '_pred_train.txt'
    out_train.to_csv(os.path.join(results_dir, fnm), float_format='%.3e')

    fnm = 'model_' + config + '_pred_test.txt'
    out_test.to_csv(os.path.join(results_dir, fnm), float_format='%.3e')

    # -----------------------------------------------------------------
    # extract model output from bottleneck layer
    # -----------------------------------------------------------------

    aux_model = Model(inputs = autoencoder.inputs,
        outputs = [autoencoder.layers[3].output])

    embedding_train = aux_model.predict(trainX)
    embedding_test  = aux_model.predict(testX)

    embedding_train_df = pd.DataFrame(data=embedding_train, index=trainX.index)
    embedding_test_df  = pd.DataFrame(data=embedding_test,  index=testX.index)

    fnm = 'model_' + config + '_bottleneck_train.txt'
    embedding_train_df.to_csv(os.path.join(results_dir, fnm), float_format='%.3e')

    fnm = 'model_' + config + '_bottleneck_test.txt'
    embedding_test_df.to_csv(os.path.join(results_dir, fnm), float_format='%.3e')

# -----------------------------------------------------------------
# parameters
# -----------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Model somatic mutation data using deep learning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--input", "-i",
    type = str,
    dest = "input_file",
    help = "input: path to input file"
)

parser.add_argument(
    "--data-dir", "-D",
    type = str,
    default = "../data",
    help = "directory where data are placed"
)
parser.add_argument(
    "--fig-dir", "-F",
    type = str,
    default = "../figures",
    help = "directory where figures are saved"
)
parser.add_argument(
    "--results-dir", "-R",
    type = str,
    default = "../results",
    help = "directory where results are saved"
)
parser.add_argument(
    "--hidden-size1", 
    type = int,
    default = 400,
    help = "number of nodes for hidden layer 1"
)
parser.add_argument(
    "--hidden-size2", 
    type = int,
    default = 200,
    help = "number of nodes for hidden layer 2"
)
parser.add_argument(
    "--n-layers", "-L",
    type = int,
    default = 1,
    help = "number of layers for encoder and decoder: 1 or 2"
)
parser.add_argument(
    "--batch-size", "-M",
    type = int,
    default = 32,
    help = "batch size used when training"
)
parser.add_argument(
    "--learn-rate",
    type = float,
    default = 1e-3,
    help = "learning rate when training"
)
parser.add_argument(
    "--n-epochs", "-e",
    type = int,
    default = 50,
    help = "number of epochs for which to train"
)


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))

