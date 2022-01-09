# Different models and hyper parameters

Taking HLA TCR association prediction as an example

## Six versions of models

### v1 basic model:

directly concatenate encoded HLA and TCR information together and pass to dense layers.

v1_basic_template_HLA_I_gpu.py
   |
   -- \_st_bpad_general_I.py
       |
       -- \_st_encode_general.py
   -- \_st_get_acc_classes_loss.py
   -- \_v1_build_model_general_u.py
   
   
### v2 additional separate dense layers:

before concatenating the HLA and TCR information together, add two separate dense layers, with one taking the HLA information as input and the other one taking the TCR information as input.

### v3 one layer CNN and additional separate dense layers:

in addition to v2, before coming to the separate dense layers, for TCR information, we apply a 1D CNN layer on encoded CDR3 amino acid sequence before concatenation. Inside the CNN layer, we use 8 filters of size 2 and the maxpooling is done with size 2 and stride 1. The output of this CNN layer, after being concatenated with other features of TCR, is taken as input to the separate dense layer for TCR.

### v4 One layer CNN plus encoded CDR3 and separate dense layers:

In addition to what is done in the v3, the output of CNN layer is concatenated not only with other features of TCR, but also with a copy of the original encoded CDR3 amino acid sequence (this is what the CNN layer takes as input as well) before being taken as input to the separate dense layer for TCR.

### v5 LSTM plus encoded CDR3 and separate dense layers:

Compared to v4, the CNN layer is replaced by a bidirectional LSTM layer. In each direction, the LSTM unit has hidden size 8. The output of the LSTM layer contains the hidden states in both directions at all locations in CDR3 amino acid sequence.

### v6 Two CNN layers plus encoded CDR3 and separate dense layers:

Compared to v4, another 1D CNN layer is added after the existing CNN layer. Inside the second CNN layer, we use 16 filters of size 2 and stride 1. This setting of two CNN layers and CNN parameters follow those as in Beshnova, Daria, et al. (2020).


## 120 combinations of hyper parameters

For each version of model, we run experiments for the 4\*2\*15=120 different parameter combinations, which involves:

### Encoding methods for amino acids (4)

We try four different ways of encoding amino acids in HLA pseudo sequence, CDR1, CDR2, CDR2.5, and CDR3. These methods are one hot encoding, BLOSUM62 matrix, Atchely factors, and PCA encoding as used in Beshnova, Daria, et al. (2020).

### Usage of V gene information (2)

We try two ways to use V gene information. One is to use the amino acid sequence of the corresponding CDR1, CDR2 and CDR2.5 regions. The other one is to use both V gene name as a categorical variable and the sequence information of the three regions.

### Number, size and dropout of dense layers after concatenation (15)

For the dense layer part after concatenating the HLA and TCR information, when using only one dense layer, we try layer sizes 16, 32, and 64; when using two dense layers we try layer size pairs [32,16] and [64,16], where the first/second number is the size for the first/second layer.

We also try using dropout rate 0.2 or 0.5, or no dropout. When only one dense layer is used, the dropout happens after the dense layer, and when two dense layers are used, the dropout happens between the two layers.






## Reference

Beshnova, Daria, et al. "De novo prediction of cancer-associated T cell receptors for noninvasive cancer detection." Science translational medicine 12.557 (2020).
