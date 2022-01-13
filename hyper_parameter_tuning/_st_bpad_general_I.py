
# encode CDR3 len to categorical
import numpy as np
import pandas as pd

import random
from random import sample, shuffle

from collections import Counter
from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder

from _st_encode_general import encode_set

# contains hard coded elements:
# 40, the length of HLA I pseudo sequence
# 65, the number of distinct V allele names
# 27, max length of cdr3 sequence
# 12, length of cdr1 sequence
# 10, length of cdr2 sequence
# 6, length of cdr25 sequence
# the chunk of code on assigning 11,037 tcrs to training, validating and testing
#   is limited to the data from using the new pvalue cutoff
def get_data(np_train, np_valid, np_test, n_fold = 1, \
             enc_method = "one_hot", rseed = 1216):
    # load HLA pseudo information
    HLA_I_pseudo = pd.read_csv("../data/step75_HLA_I_v2_pseudo_40.csv", \
                                sep=',', header= 0)
    # load TCR information
    TCR_rmat = pd.read_csv(\
     "../data/step58_public_allele_level_tcr_name.csv",
     header = 0)
    TCR_name = [v + ',' + aa for v, aa in zip(TCR_rmat.v_allele.tolist(), \
                                              TCR_rmat.amino_acids.tolist())]
    # do one_hot_encoding for cdr3 length
    TCR_aa_list = TCR_rmat.amino_acids.tolist()
    TCR_aa_lens = [len(aa) for aa in TCR_aa_list]
    unique_TCR_aa_lens = list(set(TCR_aa_lens))
    CDR3len_enc_template = \
        np.array(unique_TCR_aa_lens).reshape(len(unique_TCR_aa_lens), 1)
    CDR3len_enc = OneHotEncoder().fit(CDR3len_enc_template)
    V_seq = TCR_rmat.v_allele.tolist()
    # get the list of all V genes appeared in public tcr matrix
    V_list = list(Counter(V_seq).keys())
    V_list.sort()
    # encoder for V gene
    # hard code 65 here for the number of different v alleles from
    # TCRs in current file
    V_enc_template = np.array(V_list).reshape(65, 1)
    V_enc = OneHotEncoder().fit(V_enc_template)
    # build encoder for amino acids in HLA and CDR3
    # one_hot_encoding first
    if enc_method == "one_hot":
        AA_SYMOLS = ['A', 'R', 'N', 'D', 'C',
                    'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P',
                    'S', 'T', 'W', 'Y', 'V']
        AA_SYMOLS.sort()
        # encoder for HLA
        # pseudo sequences in the file for HLA_I alleles
        # "X" for some HLA-I, so add 'X' to the encoding template
        AA_SYMOLS_X = AA_SYMOLS + ['X']
        encode_template = np.array([[aa] * 40 for aa in AA_SYMOLS_X])
        HLA_enc = OneHotEncoder().fit(encode_template)
        # now move on to encode the v gene and CDR3 part
        #add '.' to the aa name list for the purpose of padding short CDR3 sequences
        # encoder for CDR3
        # hard code 27 here for the max length of CDR3 aa seq in current file
        AA_SYMOLS_pad = AA_SYMOLS + ['.']
        CDR3_enc_template = np.array([[aa] * 27 for aa in AA_SYMOLS_pad])
        CDR3_enc = OneHotEncoder().fit(CDR3_enc_template)
        # define a function CDR3_enc to do the encoding instead of previous encoder
        # due to the fact that the cdr2 sequence contains '*', the CDR3_enc below is
        # adjusted to add a '*' option to make it work
        # for cdr1, cdr2, cdr25 as weill
        cdr1_template = np.array([[aa] * 12 for aa in AA_SYMOLS_pad])
        cdr1_enc = OneHotEncoder().fit(cdr1_template)
        AA_SYMOLS_pad_star = AA_SYMOLS + ['.'] + ['*']
        cdr2_template = np.array([[aa] * 10 for aa in AA_SYMOLS_pad_star])
        cdr2_enc = OneHotEncoder().fit(cdr2_template)
        cdr25_template = np.array([[aa] * 6 for aa in AA_SYMOLS_pad])
        cdr25_enc = OneHotEncoder().fit(cdr25_template)
    # or blosum62
    elif enc_method == "blosum62":
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, use the blosum62 matrix with "X"
            blosum62_matrix = pd.read_csv("../data/blosum62_X.csv", \
                                    sep=',', header= 0)
            blosum62_array = np.array(blosum62_matrix)
            blosum62_dict = defaultdict(list)
            for i in range(21):
                blosum62_dict[blosum62_array[i][0]] = \
                       blosum62_array[i][1:].tolist()
            return [[blosum62_dict[aa] for aa in hla] for hla in hla_list]
        # define encoder for CDR3
        # this encoder also works for cdr1, cdr25
        # it does not work for cdr2 due to that some sequence in cdr2 contains
        # a charater "*"
        def cdr_enc(cdr_list):
            blosum62_matrix = pd.read_csv("../data/blosum62.csv", \
                                    sep=',', header= 0)
            blosum62_array = np.array(blosum62_matrix)
            blosum62_dict = defaultdict(list)
            for i in range(20):
                blosum62_dict[blosum62_array[i][0]] = \
                       blosum62_array[i][1:].tolist() + [-4]
            blosum62_dict['.'] = [-4 for _ in range(20)] + [1]
            return [[blosum62_dict[aa] for aa in cdr] for cdr in cdr_list]
        # define an encoder specifically for cdr2
        # since some sequence in cdr2 contains a charater "*"
        def cdr2_enc(cdr2_list):
            blosum62_matrix = pd.read_csv("../data/blosum62.csv", \
                                    sep=',', header= 0)
            blosum62_array = np.array(blosum62_matrix)
            blosum62_dict = defaultdict(list)
            for i in range(20):
                blosum62_dict[blosum62_array[i][0]] = \
                       blosum62_array[i][1:].tolist() + [-4]
            blosum62_dict['.'] = [-4 for _ in range(20)] + [1]
            blosum62_dict['*'] = [-4 for _ in range(20)] + [1]
            return [[blosum62_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]
    # or atchley
    elif enc_method == "atchley":
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, so add 'X' to the encoding template
            atchley_matrix = pd.read_csv("../data/Atchley_factors.csv", \
                                    sep=',', header= 0)
            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['X'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in hla] for hla in hla_list]
        # define encoder for CDR3
        # this encoder also works for cdr1, cdr25
        # it does not work for cdr2 due to that some sequence in cdr2 contains
        # a charater "*"
        def cdr_enc(cdr_list):
            atchley_matrix = pd.read_csv("../data/Atchley_factors.csv", \
                                    sep=',', header= 0)
            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['.'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in cdr] for cdr in cdr_list]
        # define an encoder specifically for cdr2
        # since some sequence in cdr2 contains a charater "*"
        def cdr2_enc(cdr2_list):
            atchley_matrix = pd.read_csv("../data/Atchley_factors.csv", \
                                    sep=',', header= 0)
            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['.'] = [0 for _ in range(5)]
            atchley_dict['*'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]
    # AAidx_PCA
    elif enc_method == 'pca':
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, so add 'X' to the encoding template
            pca_matrix = pd.read_csv("../data/AAidx_PCA.csv", \
                                    sep=',', header= 0)
            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['X'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in hla] for hla in hla_list]
        # define encoder for CDR3
        # this encoder also works for cdr1, cdr25
        # it does not work for cdr2 due to that some sequence in cdr2 contains
        # a charater "*"
        def cdr_enc(cdr_list):
            pca_matrix = pd.read_csv("../data/AAidx_PCA.csv", \
                                    sep=',', header= 0)
            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['.'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in cdr] for cdr in cdr_list]
        # define an encoder specifically for cdr2
        # since some sequence in cdr2 contains a charater "*"
        def cdr2_enc(cdr2_list):
            pca_matrix = pd.read_csv("../data/AAidx_PCA.csv", \
                                    sep=',', header= 0)
            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['.'] = [0 for _ in range(15)]
            pca_dict['*'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]
    else:
        print("Error: enc_method is a value not coded for yet.")
        return
    # move on to select the positive and negative pairs
    # load the association matrix with HLA-I only
    df_asso = pd.read_csv(\
         "../data/step77_HLA_I_associated_TCR_v_alleles.csv",\
         sep = ",", header = 0)
    # use dictionary to keep the associated pairs
    asso_dict = defaultdict(lambda: defaultdict(int))
    for tcr, hla in zip(df_asso.tcr, df_asso.hla_allele):
        asso_dict[hla][tcr] = 1
    # set random seed
    random.seed(rseed)
    # build a dict for HLA_I allele aa info
    allele_dict = defaultdict(list)
    for name, seq in zip(HLA_I_pseudo.hla.to_list(), \
                     HLA_I_pseudo.seq.to_list()):
        allele_dict[name] = list(seq)
    # generate the pair list for postive and negative
    # positive:
    positive_ori = [(tcr, hla) for tcr, hla in \
                zip(df_asso.tcr.to_list(), df_asso.hla_allele.to_list())]
    # negative:
    shuffle(TCR_name)
    nn_train = np_train * n_fold
    nn_valid = np_valid * n_fold
    nn_test  = np_test * n_fold
    # count how many positive pairs each HLA appears in
    positive_hla_list = [item[1] for item in positive_ori]
    counter_pos_hla = Counter(positive_hla_list)
    positive_hla_unique = list(counter_pos_hla.keys())
    # construct the negative pairs that we really use
    negative_asso_dict = defaultdict(lambda: defaultdict(int))
    i = 0
    # negative test
    negative_ori = []
    for hla in positive_hla_unique:
        cnt_add = 0
        while cnt_add < (counter_pos_hla[hla]*n_fold):
            i = i%len(TCR_name)
            tcr = TCR_name[i]
            if tcr not in asso_dict[hla]:
                if tcr not in negative_asso_dict[hla]:
                    negative_ori.append((tcr, hla))
                    negative_asso_dict[hla][tcr] = 1
                    cnt_add += 1
            i += 1
    # shuffle pos and neg pairs
    shuffle(positive_ori)
    shuffle(negative_ori)
    # split positive pair list
    positive_test_ori = positive_ori[:np_test]
    positive_valid_ori = positive_ori[np_test:(np_test+np_valid)]
    positive_train_ori = positive_ori[(np_test+np_valid):]
    # split negative pair list
    negative_test_ori = negative_ori[:nn_test]
    negative_valid_ori = negative_ori[nn_test:(nn_test+nn_valid)]
    negative_train_ori  = negative_ori[(nn_test+nn_valid):]
    # pass arguments to encode_set function to do encoding and get output
    if enc_method == "one_hot":
        components_train = \
            encode_set(negative_train_ori, positive_train_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, CDR3_enc, cdr1_enc, \
                       cdr2_enc, cdr25_enc,\
                       enc_method, rseed)
        components_valid = \
            encode_set(negative_valid_ori, positive_valid_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, CDR3_enc, cdr1_enc, \
                       cdr2_enc, cdr25_enc,\
                       enc_method, rseed)
        components_test = \
            encode_set(negative_test_ori, positive_test_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, CDR3_enc, cdr1_enc, \
                       cdr2_enc, cdr25_enc,\
                       enc_method, rseed)
    elif enc_method in ["blosum62", "atchley", "pca"]:
        components_train = \
            encode_set(negative_train_ori, positive_train_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, cdr_enc, cdr_enc, \
                       cdr2_enc, cdr_enc, \
                       enc_method, rseed)
        components_valid = \
            encode_set(negative_valid_ori, positive_valid_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, cdr_enc, cdr_enc, \
                       cdr2_enc, cdr_enc, \
                       enc_method, rseed)
        components_test = \
            encode_set(negative_test_ori, positive_test_ori, allele_dict, \
                       HLA_enc, V_enc, CDR3len_enc, cdr_enc, cdr_enc, \
                       cdr2_enc, cdr_enc, \
                       enc_method, rseed)
    else:
        print("Error: enc_method is a value not coded for yet.")
        return
    return (components_train, components_valid, components_test)
