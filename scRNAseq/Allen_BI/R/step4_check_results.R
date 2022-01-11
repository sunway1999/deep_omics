
library(data.table)
library(tidyr)
library(stringr)
library(ggplot2)
library(Rtsne)

theme_set(theme_classic())

# ------------------------------------------------------------------------
# check the cell cluster information
# ------------------------------------------------------------------------

clusters = readRDS("../data/final_hvg_clust.rds")
dim(clusters)
names(clusters)

clusters[1:2,c(1,52:61)]
table(clusters$cell_type)

dat = fread("../data/cts_all_but_Micro_Endo_ordered_by_annotation.txt.gz")
dim(dat)
dat[1:2,1:3]

table(names(dat)[-1] %in% clusters$sample_name)

table(clusters$cell_type)
mat1 = match(names(dat)[-1], clusters$sample_name)
table(clusters$cell_type[mat1])

# ------------------------------------------------------------------------
# Read in the losses
# ------------------------------------------------------------------------

ds_loss = fread("../auto_encoder_plots/AE_sklearn_loss.csv")
lc_loss = fread("../auto_encoder_plots/lc_AE_sklearn_loss.csv")

dim(ds_loss)
dim(lc_loss)
ds_loss[1:2,]
lc_loss[1:2,]

table(ds_loss$epoch_idx == 1:499)
table(lc_loss$epoch_idx == 1:499)

loss = merge(ds_loss, lc_loss, by="epoch_idx", 
             suffixes=c("_dense", "_local"))
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

pdf("../auto_encoder_plots/AE_vs_lc_AE_sklearn_loss.pdf", 
    width=4, height=2.7)
g1
dev.off()

# ------------------------------------------------------------------------
# Read in encoding using locally connected AE
# ------------------------------------------------------------------------

encoding = fread("../auto_encoder_plots/lc_AE_sklearn_latent_output.csv")
dim(encoding)

encoding[1:2,]

edat = data.matrix(encoding[,1:20])
summary(edat)

n_non_zero = colSums(edat > 0)
n_non_zero

edat = edat[, n_non_zero > 20]
dim(edat)

edat[edat > 4] = 4
set.seed(100)
date()
tsne = Rtsne(edat, pca = FALSE)
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

pdf("../auto_encoder_plots/lc_AE_sklearn_TSNE.pdf", width=3.5, height=2.7)
gp1
dev.off()

sessionInfo()
q(save="no")
