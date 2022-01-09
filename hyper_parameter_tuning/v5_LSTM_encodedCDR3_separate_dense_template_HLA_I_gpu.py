#!/usr/bin/env python3


import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf


from random import shuffle, sample

from collections import Counter
from collections import defaultdict

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Reshape

from numpy.random import seed

# the following line is to deal with the case that self-defined module
# in the submission directory are not found when submitted through sbatch, as in
# https://stackoverflow.com/questions/46718499/
# how-does-one-make-sure-that-the-python-submission-script-in-slurm-is-in-the-loca/
# 46724189?noredirect=1#comment80425963_46724189
sys.path.append(os.getcwd())
# this file provides evaulation functions
# which gives the weighted loss, accuracy,
# auc roc, auc pr, true positive rate and true negative rate
from _st_get_acc_classes_loss import get_acc_classes
# this file constructs and encodes the data
from _st_bpad_general_I import get_data
# this file builds the model
from _v5_build_model_general_u import get_model


def pred_asso(enc_method, n_fold, lr, V_cdrs = 2, \
              CNN_flag = False, n_grams = [1,3,5,7,9], \
              n_filters = 100, pl_size = 0, strides = 0, \
              n_dense = 1, n_units = [16], dropout_flag = False,
              p_dropout = 0.2, rseed = 1216, tf_seed = 2207):
    # setting both numpy and sensorflow random seeds to make sure of
    # the reproducibility of the code

    seed(rseed)
    tf.random.set_seed(tf_seed)

    patience = 10

    if V_cdrs == 0:
        str_V_cdrs = "_V"
    elif V_cdrs == 1:
        str_V_cdrs = "_cdrs"
    else:
        str_V_cdrs = "_V_cdrs"

    setting_name = \
     enc_method + '_len_cdr3_' + 'n_fold_' + str(n_fold) + '_' + str(lr)[2:] + \
     str_V_cdrs + ('_CNN_filters_' + str(n_filters) + \
       ('_plsize_' + str(pl_size) + '_strides_' + str(strides))*int(bool(pl_size))) * \
     int(CNN_flag) + '_dense' + str(n_dense) + \
     '_n_units_' + '_'.join([str(n) for n in n_units]) + \
      ('_dropout_p_' + str(p_dropout)[2:]) * int(dropout_flag)
    # specify the name of the model with best performance on validation data set
    # in the training process, for saving the model parameters in later code
    checkpoint_path = \
      './v5_HLA_I_saved_models/v5_HLA_I_best_valid_' + \
      setting_name + '.hdf5'

    # number of positive pairs in traning/validating/testing
    # subject to the constraint that we have 6,423 positive pairs
    # these three need to sum to 6,423 otherwise need to modify
    # the get_data function
    np_train = 3853
    np_valid = 1285
    np_test  = 1285

    ((HLA_encoded_train, V_encoded_train, CDR3_encoded_train, CDR3_len_train, \
      cdr1_encoded_train, cdr2_encoded_train, cdr25_encoded_train, y2_train),\
     (HLA_encoded_valid, V_encoded_valid, CDR3_encoded_valid, CDR3_len_valid, \
      cdr1_encoded_valid, cdr2_encoded_valid, cdr25_encoded_valid, y2_valid),\
     (HLA_encoded_test, V_encoded_test, CDR3_encoded_test, CDR3_len_test, \
      cdr1_encoded_test, cdr2_encoded_test, cdr25_encoded_test, y2_test)) = \
     get_data(np_train = np_train, np_valid = np_valid, np_test = np_test, \
              n_fold = n_fold, enc_method = enc_method, rseed = rseed)

    HLA_encoded_train.shape
    HLA_encoded_valid.shape
    HLA_encoded_test.shape
    V_encoded_train.shape
    CDR3_encoded_train.shape
    CDR3_len_train.shape
    cdr1_encoded_train.shape
    cdr2_encoded_train.shape
    cdr25_encoded_train.shape
    y2_train.shape

    # get the model
    model = get_model(HLA_shape = HLA_encoded_train.shape[1:],
                      V_shape = V_encoded_train.shape[1:],
                      CDR3_shape = CDR3_encoded_train.shape[1:],
                      len_shape = CDR3_len_train.shape[1:],
                      cdr1_shape = cdr1_encoded_train.shape[1:],
                      cdr2_shape = cdr2_encoded_train.shape[1:],
                      cdr25_shape = cdr25_encoded_train.shape[1:],
                      V_cdrs = V_cdrs,
                      CNN_flag = CNN_flag,
                      n_grams = n_grams,
                      n_filters = n_filters,
                      pl_size = pl_size,
                      strides = strides,
                      n_dense = n_dense,
                      n_units = n_units,
                      dropout_flag = dropout_flag,
                      p_dropout = p_dropout)

    model.summary()

    # compile model

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name = "accuracy"),
        tf.keras.metrics.AUC(name = "auc_roc", curve='ROC'),
        tf.keras.metrics.AUC(name = "auc_pr", curve='PR')
    ]

    adam_optim = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer= adam_optim, \
                  metrics=METRICS)

    # fit model

    weights = {0:1, 1:n_fold}

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc_roc',
                                                patience=patience,
                                                mode = 'max')
    # this check_point saves the model with best performance on validation data
    check_point = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor= 'val_auc_roc',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True
    )

    # validation sample weight seems to be needed for tensorflow version
    # on gpu on server, but not for the version on mac desktop
    y2_valid_list = [y[0] for y in y2_valid]
    w_valid_list = [1 if y == 0 else n_fold for y in y2_valid_list]

    model.fit(x = [HLA_encoded_train, V_encoded_train, CDR3_encoded_train, \
                   CDR3_len_train, cdr1_encoded_train, \
                   cdr2_encoded_train, cdr25_encoded_train], y = y2_train, \
              validation_data = ([HLA_encoded_valid, V_encoded_valid, \
              CDR3_encoded_valid, CDR3_len_valid, cdr1_encoded_valid, \
              cdr2_encoded_valid, cdr25_encoded_valid], y2_valid, np.array(w_valid_list)), \
              class_weight=weights, callbacks=[callback, check_point], \
              epochs=200, batch_size=32)

    loss_t_list = []
    acc_t_list = []
    auc_roc_t_list = []
    auc_pr_t_list = []
    acc_on_po_list = []
    acc_on_ne_list = []

    data_to_test = [HLA_encoded_test, V_encoded_test, CDR3_encoded_test, \
                    CDR3_len_test, cdr1_encoded_test, \
                    cdr2_encoded_test, cdr25_encoded_test]
    label_to_test = y2_test

    # reload the best model saved in the training process and evaluate it

    tf.keras.backend.clear_session()

    model = get_model(HLA_shape = HLA_encoded_train.shape[1:],
                      V_shape = V_encoded_train.shape[1:],
                      CDR3_shape = CDR3_encoded_train.shape[1:],
                      len_shape = CDR3_len_train.shape[1:],
                      cdr1_shape = cdr1_encoded_train.shape[1:],
                      cdr2_shape = cdr2_encoded_train.shape[1:],
                      cdr25_shape = cdr25_encoded_train.shape[1:],
                      V_cdrs = V_cdrs,
                      CNN_flag = CNN_flag,
                      n_grams = n_grams,
                      n_filters = n_filters,
                      pl_size = pl_size,
                      strides = strides,
                      n_dense = n_dense,
                      n_units = n_units,
                      dropout_flag = dropout_flag,
                      p_dropout = p_dropout)

    adam_optim = tf.keras.optimizers.Adam(learning_rate=lr)
    model.load_weights(checkpoint_path)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer= adam_optim, \
                  metrics=METRICS)

    loss_t, acc_t, auc_roc_t, auc_pr_t, acc_on_po, acc_on_ne = \
        get_acc_classes(model, data_to_test, label_to_test, n_fold)


    loss_t_list += [loss_t]
    # here the accuracy is taken to be the weighed accuracy
    # not the default one from model evaluation
    acc_t_list += [(acc_on_po+acc_on_ne)/2]
    auc_roc_t_list += [auc_roc_t]
    auc_pr_t_list += [auc_pr_t]
    acc_on_po_list += [acc_on_po]
    acc_on_ne_list += [acc_on_ne]

    # write out evaluation metrics
    df_metric = \
      pd.DataFrame(zip(loss_t_list, acc_t_list, auc_roc_t_list, auc_pr_t_list, acc_on_po_list, acc_on_ne_list), \
                   columns = ['loss','acc', 'auc_roc', 'auc_pr', 'acc_on_po', 'acc_on_ne'])
    df_metric.to_csv("./v5_HLA_I_metrics/v5_HLA_I_metrics_" + \
                      setting_name + '.csv', index = False)




if __name__ == "__main__":
    if len(sys.argv) > 16:
        #print("Error: too many arguments")
        sys.exit("Error: too many input arguments")
    print("file name is ", sys.argv[0])
    # enc_method
    enc_method = sys.argv[1]
    print("enc_method = ", enc_method)
    if enc_method not in ["one_hot", "blosum62", "atchley", "pca"]:
        sys.exit("Error: enc_method is not coded for yet.")
    # n_fold
    n_fold = int(sys.argv[2])
    print("n_fold = ", n_fold)
    # lr, learning rate
    lr = float(sys.argv[3])
    print("lr = ",lr)
    # V_cdrs, whether to use V, cdrs or both
    V_cdrs = int(sys.argv[4])
    print("V_cdrs = ", V_cdrs)
    if V_cdrs not in [0, 1, 2]:
        sys.exit("Error: V_cdrs is not on of 0, 1, 2.")
    # CNN_flag
    if sys.argv[5] == 'True':
        CNN_flag = True
    else:
        CNN_flag = False
    print("CNN_flag = ", CNN_flag)
    # n_grams
    len_n_grams_str = len(sys.argv[6])
    n_grams_str = sys.argv[6][1:len_n_grams_str-1].split(',')
    n_grams = [int(i) for i in n_grams_str]
    print("n_grams = ", n_grams)
    # n_filters
    n_filters = int(sys.argv[7])
    print("n_filters = ", n_filters)
    # pl_size
    pl_size = int(sys.argv[8])
    print("pl_size = ", pl_size)
    # strides
    strides = int(sys.argv[9])
    print("strides = ", strides)
    # n_dense
    n_dense = int(sys.argv[10])
    print("n_dense = ", n_dense)
    # n_units
    len_n_units_str = len(sys.argv[11])
    n_units_str = sys.argv[11][1:len_n_units_str-1].split(',')
    n_units = [int(i) for i in n_units_str]
    print("n_units = ", n_units)
    if len(n_units) != n_dense:
        sys.exit("Error: n_dense and n_units do not match.")
    # dropout_flag
    if sys.argv[12] == 'True':
        dropout_flag = True
    else:
        dropout_flag = False
    print("dropout_flag = ", dropout_flag)
    # p_dropout
    p_dropout = float(sys.argv[13])
    print('p_dropout = ', p_dropout)
    # rseed
    if len(sys.argv) > 14:
        rseed = int(sys.argv[14])
    else:
        rseed = 1216
    print('rseed = ', rseed)
    if len(sys.argv) > 15:
        tf_seed = int(sys.argv[15])
    else:
        tf_seed = 2207
    print('tf_seed = ', tf_seed)
    # run main prediction function
    pred_asso(enc_method, n_fold, lr, V_cdrs, CNN_flag, n_grams, n_filters, \
              pl_size, strides, \
              n_dense, n_units, dropout_flag, p_dropout, rseed, tf_seed)
