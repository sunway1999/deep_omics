
library(data.table)
library(tidyr)
library(stringr)
library(ggplot2)
library(Rtsne)
library(glmnet)
library(RDRToolbox)

theme_set(theme_classic())

# ------------------------------------------------------------------------
# check the cell cluster information
# ------------------------------------------------------------------------

clusters = readRDS("../data/final_hvg_clust.rds")
dim(clusters)
names(clusters)

clusters[1:2,c(1,52:61)]
table(clusters$cell_type)

dat = fread("../data/cts_all_but_Micro_Endo_ordered.txt.gz")
dim(dat)
dat[1:2,1:3]

table(names(dat)[-1] %in% clusters$sample_name)

table(clusters$cell_type)
mat1 = match(names(dat)[-1], clusters$sample_name)
table(clusters$cell_type[mat1])

# ------------------------------------------------------------------------
# Read in the losses
# ------------------------------------------------------------------------

ds_loss = fread("../new_plots/AE_manual_loss.csv")
lc_loss = fread("../new_plots/lc_AE_manual_loss.csv")

dim(ds_loss)
dim(lc_loss)
ds_loss[1:2,]
lc_loss[1:2,]

table(ds_loss$epoch_idx == 1:499)
table(lc_loss$epoch_idx == 1:499)

loss = merge(ds_loss, lc_loss, by="epoch_idx", suffixes=c("_dense", "_local"))
dim(loss)
loss[1:2,]

summary(loss)
loss_long = pivot_longer(loss, cols=!epoch_idx, names_to = "loss_type", 
                         values_to = "loss_value")
dim(loss_long)
loss_long[1:2,]

loss_long$data = str_extract(loss_long$loss_type, "[:alpha:]+(?=_)")
loss_long$NN   = str_extract(loss_long$loss_type, "(?<=loss_)[:alpha:]+")

dim(loss_long)
loss_long[1:2,]

g1 = ggplot(loss_long, 
            aes(x=epoch_idx, y=loss_value, color=NN, group=loss_type)) + 
  geom_line(aes(linetype=data)) + xlab("Epoch Index") + ylab("Loss")

pdf("../new_plots/AE_lc_dense_manual_loss.pdf", width=4, height=2.7)
g1
dev.off()

# ------------------------------------------------------------------------
# Read in encoding for 
# ------------------------------------------------------------------------

encoding = fread("../new_plots/lc_AE_manual_latent_output.csv")
dim(encoding)

encoding[1:2,]

set.seed(100)
date()
tsne = Rtsne(encoding[,1:20], pca = FALSE)
date()

df_tsne = data.frame(tsne$Y)
dim(df_tsne)

table(encoding$cellname %in% clusters$sample_name)
mat1 = match(encoding$cellname, clusters$sample_name)
df_tsne$cell_type = clusters$cell_type[mat1]
table(df_tsne$cell_type)

dim(df_tsne)
df_tsne[1:2,]

cols = c("#FF7F00","orchid", "red","dodgerblue2","black")

gp1 = ggplot(df_tsne, aes(X1,X2,col=cell_type)) + 
  geom_point(size=0.2,alpha=0.6) + theme_classic() + 
  scale_color_manual(values=cols) + 
  guides(color = guide_legend(override.aes = list(size=3)))

gp1

pdf("../new_plots/AE_lc_manual_TSNE.pdf", width=3.5, height=2.7)
gp1
dev.off()

# ------------------------------------------------------------------------
# prepare data
# ------------------------------------------------------------------------

dat_training = dat[, .SD, .SDcols=setdiff(names(dat), 
                                          c("gene_name", encoding$cellname))]
dim(dat_training)
dat_training[1:2,1:3]

dat_testing = dat[, .SD, .SDcols=encoding$cellname]
dim(dat_testing)
dat_testing[1:2,1:3]

rd = colSums(dat_training)
summary(rd)
dat_training = log(t(dat_training + 1)/(median(rd)/rd))

rd = colSums(dat_testing)
summary(rd)
dat_testing = log(t(dat_testing + 1)/(median(rd)/rd))

dim(dat_training)
dat_training[1:2,1:3]

dim(dat_testing)
dat_testing[1:2,1:3]

# ------------------------------------------------------------------------
# try LLE
# ------------------------------------------------------------------------

date()
lle_test = LLE(dat_testing, dim=20, k=5)
date()

dim(lle_test)

set.seed(100)
date()
tsne_lle = Rtsne(lle_test, pca = FALSE)
date()

df_tsne_lle = data.frame(tsne_lle$Y)
mat1 = match(rownames(dat_testing), clusters$sample_name)
df_tsne_lle$cell_type = clusters$cell_type[mat1]
table(df_tsne_lle$cell_type)

dim(df_tsne_lle)
df_tsne_lle[1:2,]

cols = c("#FF7F00","orchid", "red","dodgerblue2","black")

gp2 = ggplot(df_tsne_lle, aes(X1,X2,col=cell_type)) + 
  geom_point(size=0.2,alpha=0.6) + theme_classic() + 
  scale_color_manual(values=cols) + 
  guides(color = guide_legend(override.aes = list(size=3)))

gp2

pdf("../new_plots/lle_TSNE_testing.pdf", width=3.5, height=2.7)
gp2
dev.off()

# ------------------------------------------------------------------------
# try MDS
# ------------------------------------------------------------------------


date()
mds = cmdscale(dist(dat_training), k=20)
date()

dim(mds)

set.seed(100)
date()
tsne_mds = Rtsne(mds, pca = FALSE)
date()

df_tsne_mds = data.frame(tsne_mds$Y)
mat1 = match(rownames(dat_training), clusters$sample_name)
df_tsne_mds$cell_type = clusters$cell_type[mat1]
table(df_tsne_mds$cell_type)

dim(df_tsne_mds)
df_tsne_mds[1:2,]

cols = c("#FF7F00","orchid", "red","dodgerblue2","black")

gp2 = ggplot(df_tsne_mds, aes(X1,X2,col=cell_type)) + 
  geom_point(size=0.2,alpha=0.6) + theme_classic() + 
  scale_color_manual(values=cols) + 
  guides(color = guide_legend(override.aes = list(size=3)))

gp2

pdf("../new_plots/mds_TSNE_training.pdf", width=3.5, height=2.7)
gp2
dev.off()

# ------------------------------------------------------------------------
# MDS in testing data
# ------------------------------------------------------------------------

dim(mds)
mds[1:2,1:3]

mds_testing = matrix(NA, nrow=nrow(dat_testing), ncol=20)

ss = rep(NA, 20)
for(k in 1:20){
  if(k %% 2 == 0){
    cat(k, date(), "\n")
  }
  # k = 1
  y = mds[,k]
  fit_k = glmnet(dat_training, y)
  ss[k] = min(fit_k$lambda[which(fit_k$dev.ratio > 0.99)])
  beta_k = coef(fit_k, s = ss[k])[-1,1]
  mds_testing[,k] = dat_testing %*% beta_k
}

dim(mds_testing)
mds_testing[1:2,1:3]


set.seed(100)
date()
tsne_mds = Rtsne(mds_testing, pca = FALSE)
date()

df_tsne_mds = data.frame(tsne_mds$Y)
mat1 = match(rownames(dat_testing), clusters$sample_name)
df_tsne_mds$cell_type = clusters$cell_type[mat1]
table(df_tsne_mds$cell_type)

dim(df_tsne_mds)
df_tsne_mds[1:2,]

cols = c("#FF7F00","orchid", "red","dodgerblue2","black")

gp3 = ggplot(df_tsne_mds, aes(X1,X2,col=cell_type)) + 
  geom_point(size=0.2,alpha=0.6) + theme_classic() + 
  scale_color_manual(values=cols) + 
  guides(color = guide_legend(override.aes = list(size=3)))

gp3

pdf("../new_plots/mds_TSNE_testing.pdf", width=3.5, height=2.7)
gp3
dev.off()

sessionInfo()
q(save="no")
