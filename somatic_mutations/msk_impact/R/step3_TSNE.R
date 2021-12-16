
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

# ---------------------------------------------------------------------------
# read in mutation data
# ---------------------------------------------------------------------------

bn = list.files(path="../results", pattern = "_bottleneck_train.txt", 
                full.names=TRUE)
bn
bn.label = gsub("../results/", "", bn, fixed=TRUE)
bn.label = gsub("_bottleneck_train.txt", "", bn.label)

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
# check how often a bottole neck node has sd almost 0
# ---------------------------------------------------------------------------

for(k in 1:length(bn.label)){
  bnk = bn.label[k]
  dat1 = datL[[bnk]]
  dim(dat1)
  dat1[1:2,]
  apply(dat1, 2, sd)
  
  dat2 = unique(dat1)
  dim(dat1)
  dim(dat2)
  
  set.seed(100)
  date()
  tsne = Rtsne(dat2, pca = FALSE)
  date()
  
  df_tsne = data.frame(tsne$Y)
  rownames(df_tsne) = rownames(dat2)
  sampleID = gsub("-", ".", samples$SAMPLE_ID, fixed=TRUE)
  samples2 = samples[match(rownames(dat2), sampleID),]
  
  dim(df_tsne)
  df_tsne[1:2,]
  
  dim(samples2)
  samples2[1:2,]
  
  df_tsne = cbind(df_tsne, samples2)
  dim(df_tsne)
  df_tsne[1:2,]
  
  table(rownames(df_tsne) == gsub("-", ".", df_tsne$SAMPLE_ID, fixed=TRUE))
  
  top10 = sort(table(df_tsne$CANCER_TYPE), decreasing=TRUE)[1:10]
  top10
  
  df_tsne$cancerType = rep("Others", nrow(df_tsne))
  w2kp = which(df_tsne$CANCER_TYPE %in% names(top10))
  
  df_tsne$cancerType[w2kp] = df_tsne$CANCER_TYPE[w2kp]
  table(df_tsne$cancerType)
  
  
  .set_color_11 <- function() {
    myColors <- c( "dodgerblue2",
                   "green4", 
                   "#6A3D9A", # purple
                   "#FF7F00", # orange
                   "black",
                   "yellow", 
                   "tan4",
                   "grey",
                   "#FB9A99", # pink
                   "orchid",
                   "red")
    id <- sort(unique(df_tsne$cancerType))
    names(myColors)<-id
    scale_colour_manual(name = "cancer type", values = myColors)
  }
  
  
  gp1 = ggplot(df_tsne, aes(X1,X2,col=cancerType)) + 
    geom_point(size=0.8,alpha=0.7) + theme_classic() + 
    .set_color_11() + 
    guides(color = guide_legend(override.aes = list(size=3)))
  
  ggsave(sprintf("../figures/_tSNE_bottleneck/%s.png", bnk), gp1, 
         width=7, height=5, units="in")
}


sessionInfo()
q(save="no")



