
# ---------------------------------------------------------------------------
# after running auto-encoders using the code in python folder
# using this code to check the results
# ---------------------------------------------------------------------------

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

datL = list()

for(b1 in bn){
  dat1 = read.csv(b1)
  rownames(dat1) = dat1$X
  dat1 = data.matrix(dat1[,-1])
  datL[[b1]] = dat1
}

dim(datL[[1]])
datL[[1]][1:2,]

# ---------------------------------------------------------------------------
# check how often a bottle neck node has sd almost 0
# ---------------------------------------------------------------------------

fun1 <- function(x){
  length(which(apply(x, 2, sd)  < 1e-5))
}

table(sapply(datL, fun1))
sapply(datL, fun1)

# ---------------------------------------------------------------------------
# evaluate similarities across samples
# ---------------------------------------------------------------------------

ndim = ncol(datL[[1]])
ncf  = length(bn)

aveR2 = medR2 = q1R2 = q3R2 = minR2 = maxR2 = matrix(NA, nrow=ncf, ncol=ncf)

for(i in 1:ncf){
  for(j in 1:ncf){
    if(i == j) next
    
    R2ij = rep(NA, ndim)
    Xj   = datL[[j]]
    
    for(k in 1:ndim){
      yk = datL[[i]][,k]
      if(sd(yk) < 1e-5){ next }
      
      lmijk = summary(lm(yk ~ Xj))
      R2ij[k] = lmijk$r.squared
    }
    
    aveR2[i,j] = mean(R2ij, na.rm=TRUE)
    minR2[i,j] = min(R2ij,  na.rm=TRUE)
    maxR2[i,j] = max(R2ij,  na.rm=TRUE)
    
    qsij       = quantile(R2ij, probs=c(0.25, 0.5, 0.75), na.rm=TRUE)
    q1R2[i,j]  = qsij[1]
    medR2[i,j] = qsij[2]
    q3R2[i,j]  = qsij[3]
  }
}

summary(c(aveR2))
summary(c(minR2))
summary(c(q1R2))
summary(c(medR2))
summary(c(q3R2))
summary(c(maxR2))


sessionInfo()
q(save="no")



