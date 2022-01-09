# this file helps with encoding under HLA-I alleles

import numpy as np
import pandas as pd
import random

from random import shuffle
from collections import defaultdict

# the version of padding function used here follows
# bradley's aligning approach as in
# https://github.com/phbradley/conga/blob/master/conga/tcrdist/tcr_distances.py

# contains hard-coded elements:
# max length 27 for CDR3 sequence padding
# and the following two shapes under one-hot-encoding
# (40, 21), the shape of encoded HLA-I alleles
# (27, 21), the shape of encoded CDR3
# (12, 21), the shape of encoded cdr1
# (10, 22), the shape of encoded cdr2
# (6, 21), the shape of encoded cdr25
def encode_set(negative_ori, positive_ori, allele_dict, \
               HLA_enc, V_enc, CDR3len_enc, CDR3_enc, cdr1_enc, cdr2_enc, \
               cdr25_enc, enc_method = "one_hot",\
               rseed = 2134):
    # set random seed
    random.seed(rseed)
    # define padding function for later use
    def pad_cdr3(cdr3, max_length, pad_letter):
        if len(cdr3) == max_length:
            return cdr3
        else:
            gap_start = min( 6, 3 + (len(cdr3)-5)//2 )
            return cdr3[:gap_start] + [pad_letter] * (max_length - len(cdr3)) + \
                cdr3[gap_start:]
    # load info for cdr1, cdr2, cdr25 encoding
    V_info = pd.read_csv("../data/combo_xcr.tsv", sep='\t')
    V_sub_info = V_info.loc[(V_info.organism == 'human') \
                          & (V_info.chain == 'B') \
                          & (V_info.region == 'V')]
    cdr1_dict = defaultdict(str)
    cdr2_dict = defaultdict(str)
    cdr25_dict = defaultdict(str)
    for allele, cdrs in zip(V_sub_info.id.tolist(), V_sub_info.cdrs.tolist()):
        cdr_seqs = cdrs.split(";")
        if len(cdr_seqs) != 4:
            print("length of cdr sequences list is not 4")
            break
        cdr1_dict[allele] = cdr_seqs[0]
        cdr2_dict[allele] = cdr_seqs[1]
        cdr25_dict[allele]= cdr_seqs[2]
    # add the 'not_found' key
    cdr1_dict['not_found']  = ''.join(['.' for i in range(12)])
    cdr2_dict['not_found']  = ''.join(['.' for i in range(10)])
    cdr25_dict['not_found'] = ''.join(['.' for i in range(6)])
    # this file translates the v_allele name to the format in
    # "../data/combo_xcr.tsv"
    df_v_allele_trans = pd.read_csv(\
            "../data/step56_v_allele_translate_table_fillin.csv", \
            header = 0)
    v_allele_trans_dict = defaultdict(str)
    for v_allele, translate in \
        zip(df_v_allele_trans.v_allele_new.tolist(),\
            df_v_allele_trans.v_allele_translate.tolist()):
        v_allele_trans_dict[v_allele] = translate
    # encode negative part
    # HLA
    ne_HLA_aa_list = [allele_dict[name] for _, name in negative_ori]
    # V gene
    ne_V_list = [[tcr.split(",")[0]] for tcr, _ in negative_ori]
    ne_V_array = np.array(ne_V_list)
    ne_V_encoded = V_enc.transform(ne_V_array).toarray()
    # CDR3
    ne_CDR3_list = [pad_cdr3(list(tcr.split(",")[1]), 27, '.') \
                    for tcr, _ in negative_ori]
    # cdr1, cdr2, cdr25
    ne_V_allele_trans = [v_allele_trans_dict[tcr.split(",")[0]] \
                         for tcr, _ in negative_ori]
    ne_cdr1_list = [list(cdr1_dict[trans]) for trans in ne_V_allele_trans]
    ne_cdr2_list = [list(cdr2_dict[trans]) for trans in ne_V_allele_trans]
    ne_cdr25_list = [list(cdr25_dict[trans]) for trans in ne_V_allele_trans]
    # length of cdr3
    ne_CDR3_len_list = [[len(tcr.split(",")[1])] for tcr, _ in negative_ori]
    ne_CDR3_len = np.array(ne_CDR3_len_list)
    ne_CDR3len_encoded = CDR3len_enc.transform(ne_CDR3_len).toarray()
    # encode positive part
    # HLA
    po_HLA_aa_list = [allele_dict[name] for _, name in positive_ori]
    # V gene
    po_V_list = [[tcr.split(",")[0]] for tcr, _ in positive_ori]
    po_V_array = np.array(po_V_list)
    po_V_encoded = V_enc.transform(po_V_array).toarray()
    # CDR3
    po_CDR3_list = [pad_cdr3(list(tcr.split(",")[1]), 27, '.') \
                    for tcr, _ in positive_ori]
    # cdr1, cdr2, cdr25
    po_V_allele_trans = [v_allele_trans_dict[tcr.split(",")[0]] \
                         for tcr, _ in positive_ori]
    po_cdr1_list = [list(cdr1_dict[trans]) for trans in po_V_allele_trans]
    po_cdr2_list = [list(cdr2_dict[trans]) for trans in po_V_allele_trans]
    po_cdr25_list = [list(cdr25_dict[trans]) for trans in po_V_allele_trans]
    # length of cdr3
    po_CDR3_len_list = [[len(tcr.split(",")[1])] for tcr, _ in positive_ori]
    po_CDR3_len = np.array(po_CDR3_len_list)
    po_CDR3len_encoded = CDR3len_enc.transform(po_CDR3_len).toarray()
    # encoding for amino acis in HLA and CDR3
    if enc_method == "one_hot":
        # HLA in negative
        ne_HLA_aa_array = np.array(ne_HLA_aa_list)
        ne_HLA_encoded = HLA_enc.transform(ne_HLA_aa_array).toarray()
        ne_HLA_encoded = ne_HLA_encoded.reshape(len(negative_ori), 40, 21)
        # CDR3 in negative
        ne_CDR3_array = np.array(ne_CDR3_list)
        ne_CDR3_encoded = CDR3_enc.transform(ne_CDR3_array).toarray()
        ne_CDR3_encoded = ne_CDR3_encoded.reshape(len(negative_ori), 27, 21)
        # cdr1, cdr2, cdr25 in negative
        ne_cdr1_array = np.array(ne_cdr1_list)
        ne_cdr1_encoded = cdr1_enc.transform(ne_cdr1_array).toarray()
        ne_cdr1_encoded = ne_cdr1_encoded.reshape(len(negative_ori), 12, 21)
        ne_cdr2_array = np.array(ne_cdr2_list)
        ne_cdr2_encoded = cdr2_enc.transform(ne_cdr2_array).toarray()
        ne_cdr2_encoded = ne_cdr2_encoded.reshape(len(negative_ori), 10, 22)
        ne_cdr25_array = np.array(ne_cdr25_list)
        ne_cdr25_encoded = cdr25_enc.transform(ne_cdr25_array).toarray()
        ne_cdr25_encoded = ne_cdr25_encoded.reshape(len(negative_ori), 6, 21)
        # HLA in positive
        po_HLA_aa_array = np.array(po_HLA_aa_list)
        po_HLA_encoded = HLA_enc.transform(po_HLA_aa_array).toarray()
        po_HLA_encoded = po_HLA_encoded.reshape(len(positive_ori), 40, 21)
        # CDR3 in positive
        po_CDR3_array = np.array(po_CDR3_list)
        po_CDR3_encoded = CDR3_enc.transform(po_CDR3_array).toarray()
        po_CDR3_encoded = po_CDR3_encoded.reshape(len(positive_ori), 27, 21)
        # cdr1, cdr2, cdr25 in positive
        po_cdr1_array = np.array(po_cdr1_list)
        po_cdr1_encoded = cdr1_enc.transform(po_cdr1_array).toarray()
        po_cdr1_encoded = po_cdr1_encoded.reshape(len(positive_ori), 12, 21)
        po_cdr2_array = np.array(po_cdr2_list)
        po_cdr2_encoded = cdr2_enc.transform(po_cdr2_array).toarray()
        po_cdr2_encoded = po_cdr2_encoded.reshape(len(positive_ori), 10, 22)
        po_cdr25_array = np.array(po_cdr25_list)
        po_cdr25_encoded = cdr25_enc.transform(po_cdr25_array).toarray()
        po_cdr25_encoded = po_cdr25_encoded.reshape(len(positive_ori), 6, 21)
        # blosum62 and atchley
    elif enc_method in ["blosum62", "atchley", "pca"]:
        # negative
        ne_HLA_encoded = np.array(HLA_enc(ne_HLA_aa_list))
        ne_CDR3_encoded = np.array(CDR3_enc(ne_CDR3_list))
        ne_cdr1_encoded = np.array(cdr1_enc(ne_cdr1_list))
        ne_cdr2_encoded = np.array(cdr2_enc(ne_cdr2_list))
        ne_cdr25_encoded = np.array(cdr25_enc(ne_cdr25_list))
        # positive
        po_HLA_encoded = np.array(HLA_enc(po_HLA_aa_list))
        po_CDR3_encoded = np.array(CDR3_enc(po_CDR3_list))
        po_cdr1_encoded = np.array(cdr1_enc(po_cdr1_list))
        po_cdr2_encoded = np.array(cdr1_enc(po_cdr2_list))
        po_cdr25_encoded = np.array(cdr1_enc(po_cdr25_list))
    else:
        print("Error: enc_method is a value not coded for yet.")
        return
    # concatenate together
    HLA_encoded = np.concatenate((ne_HLA_encoded, po_HLA_encoded), axis = 0)
    V_encoded = np.concatenate((ne_V_encoded, po_V_encoded), axis = 0)
    CDR3_encoded = np.concatenate((ne_CDR3_encoded, po_CDR3_encoded), axis = 0)
    CDR3len_encoded = np.concatenate((ne_CDR3len_encoded, po_CDR3len_encoded), axis = 0)
    cdr1_encoded = np.concatenate((ne_cdr1_encoded, po_cdr1_encoded), axis = 0)
    cdr2_encoded = np.concatenate((ne_cdr2_encoded, po_cdr2_encoded), axis = 0)
    cdr25_encoded = np.concatenate((ne_cdr25_encoded, po_cdr25_encoded), axis = 0)
    # y label
    ne_y = [0 for _ in range(len(negative_ori))]
    po_y = [1 for _ in range(len(positive_ori))]
    y = np.array(ne_y + po_y).reshape(len(negative_ori) + \
                                              len(positive_ori),1)
    # shuffle
    index_list = list(range(len(negative_ori) + len(positive_ori)))
    shuffle(index_list)
    HLA_encoded = HLA_encoded[index_list]
    V_encoded = V_encoded[index_list]
    CDR3_encoded = CDR3_encoded[index_list]
    CDR3len_encoded = CDR3len_encoded[index_list]
    cdr1_encoded = cdr1_encoded[index_list]
    cdr2_encoded = cdr2_encoded[index_list]
    cdr25_encoded = cdr25_encoded[index_list]
    y = y[index_list]
    # return encoded data
    return HLA_encoded, V_encoded, CDR3_encoded, CDR3len_encoded, \
           cdr1_encoded, cdr2_encoded, cdr25_encoded, y
