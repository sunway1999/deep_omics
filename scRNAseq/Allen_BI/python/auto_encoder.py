## code written by Si Liu and Wei Sun

import pandas as pd
import numpy as np

import gzip
import os
import random
import math

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, LocallyConnected1D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from datetime import datetime


import matplotlib
matplotlib.use("Agg") # use a non-interactive backend
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt


def plot_loss(m1, start, file_name, df_filename):
    # plot the training loss and accuracy
    end = len(m1.history['loss'])
    N   = np.arange(start, end)
    s   = slice(start,end)
    #
    plt.style.use("ggplot")
    plt.figure(figsize=(5, 4))
    #
    plt.plot(N, (m1.history["loss"][s]), label="train_loss")
    plt.plot(N, (m1.history["val_loss"][s]), label="val_loss")
    #
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name)
    plt.clf()
    #
    epoch_idx = list(N)
    train_loss = m1.history["loss"][s]
    valid_loss = m1.history["val_loss"][s]
    df_loss = pd.DataFrame(list(zip(epoch_idx, train_loss, valid_loss)),
                           columns = ['epoch_idx', 'train_loss', 'valid_loss'])
    df_loss.to_csv(df_filename, index = False)

# regular autoencoder
def get_regular_AE(input_shape1, latent_size = 20, hidden_sizes = [71]):
    tf.keras.backend.clear_session()
    input_e = Input(shape=(input_shape1,))
    encoded = Dense(hidden_sizes[0], activation='relu')(input_e)
    encoded = Dense(latent_size, activation='relu')(encoded)
    decoded = Dense(hidden_sizes[0], activation='relu')(encoded)
    decoded = Dense(input_shape1, activation='sigmoid')(decoded)
    #
    AE = Model(input_e, decoded)
    return AE

# an autoencoder using locally connected layer
def get_lc_AE(input_shape1, latent_size = 20, n_filter = 1, kernel_size = 100,\
              strides = 20, hidden_sizes = [71]):
    tf.keras.backend.clear_session()
    input_e = Input(shape=(input_shape1,))
    rslayer = Reshape((input_shape1, 1), \
                       input_shape = (input_shape1, ))(input_e)
    encoded = LocallyConnected1D(filters = n_filter, kernel_size = kernel_size, \
                                 strides = strides, \
                                 activation='relu')(rslayer)
    rslayer = Reshape((hidden_sizes[0],), \
                       input_shape = (hidden_sizes[0], 1))(encoded)
    encoded = Dense(latent_size, activation='relu')(rslayer)
    decoded = Dense(hidden_sizes[0], activation='relu')(encoded)
    decoded = Dense(input_shape1, activation='sigmoid')(decoded)
    #
    lc_AE = Model(input_e, decoded)
    return lc_AE

# model_type = "lc_AE"
# split_method = "sklearn"
# number_of_epochs = 500
# batch_size = 32
# learning_rate = 1e-3
# testing_fraction = 0.2

def main(model_type = "lc_AE", split_method = "sklearn",
         number_of_epochs = 500, batch_size = 32,
         learning_rate = 1e-3, testing_fraction = 0.2):

    data_file = "../data/cts_all_but_Micro_Endo_ordered_by_annotation.txt.gz"
    data = pd.read_csv(data_file, compression='gzip', sep=",", header=0)
    data.shape
    data.iloc[:6, :5]

    gene_names = data["gene_name"]
    len(gene_names)

    col_names = list(data.columns)[1:]
    len(col_names)

    values = data.iloc[0: , 1:]
    data_array = values.to_numpy()
    data_array[:7, :5]
    # this is an arbitray choice to use top 1500 genes
    # to make it easier to generate the autoencoder
    # so 19 genes are skipped 
    data_array = data_array[:1500, ]
    data_array = data_array.T


    # transform gene expression
    depth = np.sum(data_array,1)
    depth.shape
    pd.Series(depth).describe()

    data_normalized = (data_array.T/depth).T
    print('data after normalizing by depth:')
    data_normalized.shape
    print('summation by sample:')
    pd.Series(np.sum(data_normalized,1)).describe()
    print('')


    data_normalized = normalize(data_normalized, norm='max', axis=0)

    print('check data_normalized')
    print(data_normalized.shape)
    print(data_normalized[0:7,0:5])
    print('summary of maximum per gene')
    print(pd.Series(np.amax(data_normalized, 0)).describe())
    print('summary of maximum per sample')
    print(pd.Series(np.amax(data_normalized, 1)).describe())

    train_size = math.floor((1 - testing_fraction) * data_normalized.shape[0])

    if split_method == "manual":
        random.seed(1243)
        index_list = list(range(data_normalized.shape[0]))
        random.shuffle(index_list)
        #
        trainX = data_normalized[index_list[:train_size], ]
        testX = data_normalized[index_list[train_size:], ]
        trainX.shape
        testX.shape
        #
        train_cellname = [col_names[i] for i in index_list[:train_size]]
        test_cellname = [col_names[i] for i in index_list[train_size:]]
    else:
        # sklearn style data splitting
        data_label = np.array(col_names)
        trainX, testX = train_test_split(data_normalized,
               test_size=testing_fraction, random_state=1999)
        train_cellname, test_cellname = train_test_split(data_label,
               test_size=testing_fraction, random_state=1999)


    print('training and testing data dimension:')
    print(trainX.shape)
    print(testX.shape)
    print('testX[0:2,0:9]:')
    print(testX[0:2,0:9])

    # get model
    if model_type == "AE":
        cur_model = get_regular_AE(input_shape1 = trainX.shape[1])
    else:
        cur_model = get_lc_AE(input_shape1 = trainX.shape[1])

    cur_model.summary()

    adam1 = optimizers.Adam(lr=learning_rate)

    cur_model.compile(optimizer=adam1, loss='mean_absolute_error')

    # model training
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    m1 = cur_model.fit(trainX, trainX,
                    epochs=number_of_epochs,
                    batch_size=batch_size,
                    verbose=0,
                    validation_data=(testX, testX))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time) # 29 minutes

    # write out plots for loss
    plot_path  = os.path.join('../auto_encoder_plots',
                        model_type + '_' + split_method + '_track_loss.png')
    df_path = os.path.join('../auto_encoder_plots',
                        model_type + '_' + split_method + '_loss.csv')

    plot_loss(m1, 1, plot_path, df_path)

    # write out test correlation
    test_pred  = cur_model.predict(testX)
    #rho_train = np.zeros(trainX.shape[1])
    rho_test  = np.zeros(testX.shape[1])

    for i in range(0,testX.shape[1]):
        #rho, pval = spearmanr(trainX[:,i], train_pred[:,i])
        #rho_train[i] = rho
        rho, pval = spearmanr(testX[:,i], test_pred[:,i])
        rho_test[i] = rho

    print('spearman gene-by-gene correlation for testing data:')
    print(pd.Series(rho_test).describe())

    df_rho_test = pd.DataFrame(rho_test, columns = ['rho_test'])
    df_rho_test.to_csv('../auto_encoder_plots/' + model_type + '_' +\
                        split_method + '_rho_test.csv', index = False)

    if model_type == "AE":
        aux_model = tf.keras.Model(inputs = cur_model.inputs,
                                   outputs = [cur_model.layers[2].output])
    else:
        aux_model = tf.keras.Model(inputs = cur_model.inputs,
                                   outputs = [cur_model.layers[4].output])

    intermediate_layer_output = aux_model.predict(testX)
    print('shape of latent output:')
    print(intermediate_layer_output.shape)

    df_latent = pd.DataFrame(intermediate_layer_output,
                             columns = ['d'+str(i+1) for i in range(20)])
    df_latent['cellname'] = test_cellname


    df_latent.to_csv(\
    '../auto_encoder_plots/'+model_type+'_'+split_method+'_latent_output.csv', index = False)


if __name__ == '__main__':
    main(model_type = "AE", split_method = "sklearn")
    main(model_type = "lc_AE", split_method = "sklearn")
    # main(model_type = "AE", split_method = "manual")
    # main(model_type = "lc_AE", split_method = "manual")
