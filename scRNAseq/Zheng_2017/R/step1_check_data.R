
library(Matrix)
library(fread)

# ---------------------------------------------------------------------------
# read in gene expression data
# ---------------------------------------------------------------------------

treg = fread("regulatory_t_filtered_gene_bc_matrices.tar.gz")
sessionInfo()
q(save="no")

