# compared to _v1_build_model_general_u.py,
# this version adds one dense layer after HLA
# and one dense layer after CDR3 before
# concatenating them together

from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Reshape, Dropout, concatenate


# structure currently limited to maximum two dense layers
# and one dropout layer
def get_model(HLA_shape, V_shape, CDR3_shape, len_shape, \
              cdr1_shape, cdr2_shape, cdr25_shape,
              V_cdrs = 2, \
              CNN_flag = False, n_grams = [3, 5], n_filters = 100,\
              pl_size = 0, strides = 0, \
              n_dense = 1, n_units = [16], \
              dropout_flag = False, p_dropout = 0.2):
    # check the inputs:
    if n_dense >2:
        print("Error from func get_model: number of dense layers not coded for yet.")
        return
    if n_dense > 1 and n_dense > len(n_units):
        print('Error from func get_model: n_units input is not long enough.')
        return
    if pl_size != 0:
        if strides == 0:
            print('Error from max pooling parameter setting:')
            print('If pl_size is not 0, strides must be greater than 0.')
            return
    # Define input layers
    HLA_input = Input(HLA_shape)
    HLA_reshape = Reshape((HLA_shape[0] * HLA_shape[1],), \
                           input_shape = HLA_shape)(HLA_input)
    V_input = Input(V_shape) #(28,)
    CDR3_input = Input(CDR3_shape)
    len_input = Input(len_shape)
    cdr1_input = Input(cdr1_shape)
    cdr2_input = Input(cdr2_shape)
    cdr25_input = Input(cdr25_shape)
    cdr1_reshape = Reshape((cdr1_shape[0] * cdr1_shape[1],), \
                            input_shape = cdr1_shape)(cdr1_input)
    cdr2_reshape = Reshape((cdr2_shape[0] * cdr2_shape[1],), \
                            input_shape = cdr2_shape)(cdr2_input)
    cdr25_reshape = Reshape((cdr25_shape[0] * cdr25_shape[1],), \
                            input_shape = cdr25_shape)(cdr25_input)
    # whether to use CNN or not
    if CNN_flag:
        # construct CDR3_branches
        CDR3_branches = []
        for n in n_grams:
            CDR3_branch = Conv1D(filters=n_filters, kernel_size=n, activation=relu, \
                        input_shape = CDR3_shape, name='Conv_CDR3_'+str(n))(CDR3_input)
            if pl_size == 0:
                CDR3_branch = MaxPooling1D(pool_size=27-n+1, strides=None, padding='valid', \
                                           name='MaxPooling_CDR3_'+str(n))(CDR3_branch)
            else:
                CDR3_branch = MaxPooling1D(pool_size=pl_size, strides=strides, padding='valid', \
                                           name='MaxPooling_CDR3_'+str(n))(CDR3_branch)
            CDR3_branch = Flatten(name='Flatten_CDR3_'+str(n))(CDR3_branch)
            CDR3_branches.append(CDR3_branch)
        CDR3_inter_layer = concatenate(CDR3_branches, axis=-1)
    else:
        CDR3_inter_layer = Reshape((CDR3_shape[0] * CDR3_shape[1],), \
                               input_shape = CDR3_shape)(CDR3_input)
    # concatenate four parts together
    HLA_part = Dense(64, activation = relu)(HLA_reshape)
    if V_cdrs == 2:
        TCR_combined = concatenate([V_input, len_input, CDR3_inter_layer, \
                                    cdr1_reshape, cdr2_reshape, cdr25_reshape])
        TCR_part = Dense(64, activation = relu)(TCR_combined)
        inter_layer = concatenate([HLA_part, TCR_part])
    elif V_cdrs == 0:
        TCR_combined = concatenate([V_input, len_input, CDR3_inter_layer])
        TCR_part = Dense(64, activation = relu)(TCR_combined)
        inter_layer = concatenate([HLA_part, TCR_part])
    else:
        TCR_combined = concatenate([len_input, CDR3_inter_layer, \
                                    cdr1_reshape, cdr2_reshape, cdr25_reshape])
        TCR_part = Dense(64, activation = relu)(TCR_combined)
        inter_layer = concatenate([HLA_part, TCR_part])
    # move on to see how many dense layers we want
    # and whether we want a dropout layer
    if n_dense == 1:
        if not dropout_flag:
            last_layer = Dense(n_units[0], activation = relu)(inter_layer)
        else:
            dense_layer = Dense(n_units[0], activation = relu)(inter_layer)
            last_layer = Dropout(p_dropout)(dense_layer)
    else:
        if not dropout_flag:
            first_dense = Dense(n_units[0], activation = relu)(inter_layer)
            last_layer = Dense(n_units[1], activation = relu)(first_dense)
        else:
            first_dense = Dense(n_units[0], activation = relu)(inter_layer)
            dropout_layer = Dropout(p_dropout)(first_dense)
            last_layer = Dense(n_units[1], activation = relu)(dropout_layer)
    # final output layer
    output = Dense(1, activation = 'sigmoid', name = 'output')(last_layer)
    # build the model
    model = Model(inputs=[HLA_input, V_input, CDR3_input, len_input, \
                          cdr1_input, cdr2_input, cdr25_input], outputs = output)
    return model
