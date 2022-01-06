
library(Rtsne)
library(ggplot2)

# ---------------------------------------------------------------------------
# read in clinical information per sample
# ---------------------------------------------------------------------------

samples = read.table("../msk_impact_2017/data_clinical_sample.txt", 
                     header=TRUE, sep="\t", as.is=TRUE, quote="",
                     comment.char = "", skip=4)
dim(samples)
head(samples)

table(samples$acronym)
sort(table(samples$tumor_tissue_site), decreasing=TRUE)[1:20]


table(samples$CANCER_TYPE)

# ---------------------------------------------------------------------------
# read in mutation information
# ---------------------------------------------------------------------------

mutDat = read.table("../data/mut_matrix_340_x_9112.txt", 
                    header=TRUE, sep="\t", as.is=TRUE, quote="",
                    comment.char = "")
dim(mutDat)
mutDat[1:2,1:5]

mutDat = data.matrix(mutDat[,-1])
mb = colSums(mutDat)
length(mb)
mb[1:5]
summary(mb)

# ---------------------------------------------------------------------------
# read in mutation encoding data
# ---------------------------------------------------------------------------

bn = list.files(path="../results", pattern = "_bottleneck_test.txt", 
                full.names=TRUE)
bn
bn.label = gsub("../results/", "", bn, fixed=TRUE)
bn.label = gsub("_bottleneck_test.txt", "", bn.label)

datL = list()

for(k in 1:length(bn)){
  dat1 = read.csv(bn[k])
  rownames(dat1) = dat1$X
  dat1 = data.matrix(dat1[,-1])
  datL[[bn.label[k]]] = dat1
}

dim(datL[[1]])
datL[[1]][1:2,]

# ---------------------------------------------------------------------------
# check one setup, remove the latent representation with zero weights
# ---------------------------------------------------------------------------

k = 3

bnk = bn.label[k]
bn.label
bnk

dat1 = datL[[bnk]]
dim(dat1)
dat1[1:2,]
apply(dat1, 2, sd)
w2kp = which(apply(dat1, 2, sd) > 0)
w2kp

dat2 = unique(dat1[,w2kp])
dim(dat1)
dim(dat2)

# ---------------------------------------------------------------------------
# run TSNE using bottleneck layer or PCs
# ---------------------------------------------------------------------------

set.seed(100)
date()
tsne = Rtsne(dat2, pca = FALSE)
date()

table(rownames(dat2) %in% colnames(mutDat))
mutDat.test = mutDat[,match(rownames(dat2), colnames(mutDat))]
dim(mutDat.test)
mutDat.test[1:2,1:5]

date()
tsne2 = Rtsne(t(mutDat.test), pca = TRUE, initial_dim=ncol(dat2))
date()

# ---------------------------------------------------------------------------
# construct a data frame holding the tSNE results and sample information
# ---------------------------------------------------------------------------

df_tsne = data.frame(tsne$Y)
rownames(df_tsne) = rownames(dat2)
sampleID = gsub("-", ".", samples$SAMPLE_ID, fixed=TRUE)
samples2 = samples[match(rownames(dat2), sampleID),]

dim(df_tsne)
df_tsne[1:2,]

df_tsne = cbind(df_tsne, tsne2$Y)
names(df_tsne)[3:4] = c("X1.PCA", "X2.PCA")
dim(df_tsne)
df_tsne[1:2,]

dim(samples2)
samples2[1:2,1:10]

df_tsne = cbind(df_tsne, samples2)
table(rownames(df_tsne) %in% names(mb))
df_tsne$mb = mb[match(rownames(df_tsne), names(mb))]
dim(df_tsne)
df_tsne[1:2,]

# ---------------------------------------------------------------------------
# setup color scheme
# ---------------------------------------------------------------------------

top10 = sort(table(df_tsne$CANCER_TYPE), decreasing=TRUE)[1:10]
top10

df_tsne$cancerType = rep("Others", nrow(df_tsne))
w2kp = which(df_tsne$CANCER_TYPE %in% names(top10))

df_tsne$cancerType[w2kp] = df_tsne$CANCER_TYPE[w2kp]
table(df_tsne$cancerType)

.set_color_11 <- function() {
  myColors <- c( "dodgerblue2",
                 "green4", 
                 "black",
                 "#6A3D9A", # purple
                 "#FF7F00", # orange
                 "yellow", 
                 "tan4",
                 "#FB9A99", # pink
                 "grey",
                 "orchid",
                 "red")
  id <- sort(unique(df_tsne$cancerType))
  names(myColors)<-id
  scale_colour_manual(name = "cancer type", values = myColors)
}

# ---------------------------------------------------------------------------
# plot it
# ---------------------------------------------------------------------------

gp1 = ggplot(df_tsne, aes(X1,X2,col=cancerType)) + 
  geom_point(size=0.8,alpha=0.7) + .set_color_11() + 
  guides(color = guide_legend(override.aes = list(size=3)))

gp2 = ggplot(df_tsne, aes(X1.PCA,X2.PCA,col=cancerType)) + 
  geom_point(size=0.8,alpha=0.7) + .set_color_11() + 
  guides(color = guide_legend(override.aes = list(size=3)))

ggsave(sprintf("../figures/test_%s.png", bnk), gp1, 
       width=4.9, height=2.9, units="in")

ggsave(sprintf("../figures/test_%s_PCs.png", bnk), gp2, 
       width=4.9, height=2.9, units="in")

# ---------------------------------------------------------------------------
# plot based on mutation burden
# ---------------------------------------------------------------------------

gp1 = ggplot(df_tsne, aes(X1,X2,col=log10(mb+1))) + 
  geom_point(size=0.8,alpha=0.7) + 
  scale_colour_gradientn(colours = terrain.colors(10)[1:8]) +
  guides(color = guide_legend(override.aes = list(size=3)))

gp2 = ggplot(df_tsne, aes(X1.PCA,X2.PCA,col=log10(mb+1))) + 
  geom_point(size=0.8,alpha=0.7) + 
  scale_colour_gradientn(colours = terrain.colors(10)[1:8]) +
  guides(color = guide_legend(override.aes = list(size=3)))

ggsave(sprintf("../figures/test_%s_mb.png", bnk), gp1, 
       width=4.3, height=2.9, units="in")

ggsave(sprintf("../figures/test_%s_mb_PCs.png", bnk), gp2, 
       width=4.3, height=2.9, units="in")

sessionInfo()
q(save="no")



