
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import os
import re
import random
import math

import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

# --------------------------------------------------
# read in data
# --------------------------------------------------

d = '../data/Zheng_2017'
dirs = [os.path.join(d, o, "hg19") for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
dirs

cell_types = [re.findall("../data/Zheng_2017/(\S+)_filtered", str1)[0] for str1 in dirs]
cell_types


adata = [sc.read_10x_mtx(d1, var_names='gene_symbols', cache=False) for d1 in dirs]
adata

sc9 = ad.concat(adata, label="cell_type", keys=cell_types, index_unique="_")
sc9.obs
sc9.obs.value_counts()

# --------------------------------------------------
# perform QCs
# --------------------------------------------------

sc9.var["mito"] = sc9.var_names.str.startswith("MT-")
sc9.var
sc9.var.value_counts()

sc.pp.calculate_qc_metrics(sc9, qc_vars=["mito"], 
    percent_top=None, log1p=False, inplace=True)
sc9

plt.hist(np.log10(sc9.obs["n_genes_by_counts"]), color = "skyblue", 
    ec="darkblue", bins=50)
plt.axvline(x=np.log10(200), color="grey")
plt.axvline(x=np.log10(300), color="black")
plt.show() 

plt.hist(sc9.obs["pct_counts_mito"], color = "skyblue", 
    ec="darkblue", bins=50)
plt.show() 

sc.pl.scatter(sc9, x='total_counts', y='pct_counts_mito')
sc.pl.scatter(sc9, x='total_counts', y='n_genes_by_counts')

plt.hist(np.log10(sc9.var["n_cells_by_counts"] + 1), color = "skyblue", 
    ec="darkblue", bins=50)
plt.axvline(x=np.log10(5 + 1), color="grey")
plt.axvline(x=np.log10(10 + 1), color="black")
plt.show() 

sc.pl.highest_expr_genes(sc9, n_top=20, )

# --------------------------------------------------
# filter out cells with less than 200 genes expressed 
# or with more than 5% mito expression
# --------------------------------------------------

sc.pp.filter_cells(sc9, min_genes=200)
sc9

sc9 = sc9[sc9.obs.pct_counts_mito < 5]
sc9

sc9.obs["cell_type"].value_counts()

# --------------------------------------------------
# split data to two matrices of equal size
# --------------------------------------------------

random.seed(1243)
index_list = list(range(sc9.n_obs))
random.shuffle(index_list)

train_size = math.floor(0.5*sc9.n_obs)
train_size

sc9_train = sc9[index_list[:train_size], ]
sc9_test  = sc9[index_list[train_size:], ]
sc9_train.shape
sc9_test.shape

sc9_train.write("../data/Zheng_2017/sc9_train.h5ad", compression="gzip")
sc9_test.write("../data/Zheng_2017/sc9_test.h5ad", compression="gzip")


