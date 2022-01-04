
library(GOSemSim)
library(data.table)
library(reshape2)
library(ggplot2)
library(fgsea)
library(ggpointdensity)
library(viridis)

# -------------------------------------------------------------------------
# read in gene information
# -------------------------------------------------------------------------

genes = as.data.frame(fread("../data/gene_info.txt.gz"))
dim(genes)
genes[1:2, c(1:6, ncol(genes))]
names(genes)

length(unique(genes$gene))
length(unique(genes$entrez_id))
length(unique(genes$ensembl_gene_id))

# ---------------------------------------------------------------------------
# calculate gene-gene similarity
# ---------------------------------------------------------------------------

hsGO = godata('org.Hs.eg.db', keytype = "SYMBOL", ont="BP", computeIC=FALSE) 

str(hsGO)
dim(hsGO@geneAnno); hsGO@geneAnno[1:10,]
apply(hsGO@geneAnno,2,function(xx) length(unique(xx)))

w2kp = genes$gene %in% hsGO@geneAnno$SYMBOL
table(w2kp)

genes2use = genes$gene[which(w2kp)]

go_similarity_file = "../data/gene_similarity_go.rds"

if(file.exists(go_similarity_file)){
  gS = readRDS(go_similarity_file)
}else{
  gS = mgeneSim(genes2use, semData=hsGO, measure="Wang", 
                combine="BMA", verbose=TRUE)
  saveRDS(gS, file=go_similarity_file)
}

dim(gS)
gS[1:2,1:2]

gSv = as.numeric(gS[upper.tri(gS)])
summary(gSv)

pdf("../figures/hist_gene_gene_similarity_go.pdf", width=4, height=3)
par(mar=c(5,4,1,1))
hist(gSv, main="", xlab="gene-gene similarity by GO terms")
dev.off()


# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

cormat <- reorder_cormat(gS)
upper_tri <- get_upper_tri(cormat[1:500,1:500])
melted_cormat <- melt(upper_tri, na.rm = TRUE)

go_heat_all <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + theme(axis.title.x=element_blank(),
                          axis.text.x=element_blank(),
                          axis.ticks.x=element_blank(), 
                          axis.title.y=element_blank(),
                          axis.text.y=element_blank(),
                          axis.ticks.y=element_blank())

pdf("../figures/gene_gene_similarity_go.pdf", width=8, height=7)
print(go_heat_all)
dev.off()

upper_tri <- get_upper_tri(cormat[401:420,401:420])
melted_cormat <- melt(upper_tri, na.rm = TRUE)

go_heat_20 <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1))+
  coord_fixed() + theme(axis.title.x=element_blank(),
                        axis.title.y=element_blank())

pdf("../figures/gene_gene_similarity_go_20.pdf", width=4, height=3)
print(go_heat_20)
dev.off()

# ---------------------------------------------------------------------------
# load pathway data
# ---------------------------------------------------------------------------

gmtfile_reactome  = "../../gene_annotation/c2.cp.reactome.v7.1.symbols.gmt"
pathways_reactome = gmtPathways(gmtfile_reactome)
class(pathways_reactome)
length(pathways_reactome)
summary(sapply(pathways_reactome, length))

ugenes = sort(unique(unlist(pathways_reactome)))
length(ugenes)

tgenes = table(unlist(pathways_reactome))
sort(tgenes, decreasing=TRUE)[1:50]
summary(as.numeric(tgenes))

gene2path = matrix(0, nrow=length(ugenes), ncol=length(pathways_reactome))
rownames(gene2path) = ugenes
colnames(gene2path) = names(pathways_reactome)

for(k in 1:length(pathways_reactome)){
  pk = pathways_reactome[[k]]
  gene2path[which(ugenes %in% pk),k] = 1
}

dim(gene2path)
gene2path[1:2,1:3]

table(genes2use %in% ugenes)

gene2path = gene2path[which(ugenes %in% genes2use),]
dim(gene2path)
gene2path[1:2,1:3]

NN = nrow(gene2path)
ugenes = rownames(gene2path)
table(colSums(gene2path) == sapply(pathways_reactome, length))

# ---------------------------------------------------------------------------
# calculate gene-gene similarity based on pathway data
# ---------------------------------------------------------------------------

react_similarity_file = "../data/gene_gene_pval_react.rds"

if(file.exists(react_similarity_file)){
  gReact = readRDS("../data/gene_gene_pval_react.rds")
}else{
  gReact = matrix(NA, nrow=length(genes2use), ncol=length(genes2use))
  rownames(gReact) = colnames(gReact) = genes2use
  
  for(i in 1:(length(genes2use)-1)){
    
    if(i %% 100 == 0){ cat(i, date(), "\n")}
    g1 = genes2use[i]
    if(! g1 %in% ugenes){ next }
    g1_idx = which(ugenes == g1)
    
    for(j in (i+1):length(genes2use)){
      g2 = genes2use[j]
      if(! g2 %in% ugenes){ next }
      g2_idx = which(ugenes == g2)
      
      idx_path = which(colSums(gene2path[c(g1_idx,g2_idx),])==2)
      num_path = length(idx_path)
      
      if(num_path > 0 ){
        idx_gene = rowSums(gene2path[,idx_path,drop = FALSE]) == num_path
        KK = sum(idx_gene)
      }else{
        KK = 0
      }
      
      if( KK < 2 ){
        if(KK == 0){prob = 1} else{stop("unexpected value for KK")}
      } else {
        prob = dhyper(
          x = 2,       # num white balls drawn
          m = KK,      # num white balls in urn
          n = NN - KK, # num black balls in urn
          k = 2        # num balls drawn
        )
      }
      
      gReact[i,j] = gReact[j,i] = prob
    }
  }
  
  saveRDS(gReact, file="../data/gene_gene_pval_react.rds")
}

dim(gReact)
gReact[1:5,1:4]

gS_react = 1 - gReact
diag(gS_react) = 1.0
dim(gS_react)
gS_react[1:5,1:4]

gSv_react = as.numeric(gS_react[upper.tri(gS_react)])
summary(gSv_react)

pdf("../figures/hist_gene_gene_similarity_reactome.pdf", width=4, height=3)
par(mar=c(5,4,1,1))
hist(gSv_react, main="", xlab="gene-gene similarity by Reactome pathways")
dev.off()

gS_react0 = gS_react
gS_react0[which(is.na(gS_react))] = 0

cormat <- reorder_cormat(gS_react0)
upper_tri <- get_upper_tri(cormat[501:1000,501:1000])
melted_cormat <- melt(upper_tri, na.rm = TRUE)
dim(melted_cormat)
melted_cormat[1:4,]
summary(melted_cormat$value)

reactome_heat_all <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + theme(axis.title.x=element_blank(),
                          axis.text.x=element_blank(),
                          axis.ticks.x=element_blank(), 
                          axis.title.y=element_blank(),
                          axis.text.y=element_blank(),
                          axis.ticks.y=element_blank())

pdf("../figures/gene_gene_similarity_rectome.pdf", width=8, height=7)
print(reactome_heat_all)
dev.off()

upper_tri <- get_upper_tri(cormat[501:520,501:520])
melted_cormat <- melt(upper_tri, na.rm = TRUE)

reactome_heat_20 <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1))+
  coord_fixed() + theme(axis.title.x=element_blank(),
                        axis.title.y=element_blank())

pdf("../figures/gene_gene_similarity_rectome_20.pdf", width=4, height=3)
print(reactome_heat_20)
dev.off()

# ---------------------------------------------------------------------------
# combine gene-gene similarity
# ---------------------------------------------------------------------------

dim(gS)
dim(gS_react)
gS[1:3,1:3]
gS_react[1:3,1:3]

table(rownames(gS) %in% rownames(gS_react))
mat1 = match(rownames(gS), rownames(gS_react))

gS_combine = 0.5*(gS + gS_react[mat1,mat1])
wNA = which(is.na(gS_react[mat1,mat1]), arr.ind=TRUE)
gS_combine[wNA] = gS[wNA]

gSv_combine = as.numeric(gS_combine[upper.tri(gS_combine)])
summary(gSv_combine)

pdf("../figures/hist_gene_gene_similarity_combine.pdf", width=4, height=3)
par(mar=c(5,4,1,1))
hist(gSv_combine, main="", xlab="combined gene-gene similarity")
dev.off()

cormat <- reorder_cormat(gS_combine)
upper_tri <- get_upper_tri(cormat[501:1000,501:1000])
melted_cormat <- melt(upper_tri, na.rm = TRUE)
dim(melted_cormat)
melted_cormat[1:4,]
summary(melted_cormat$value)

heat_all <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + theme(axis.title.x=element_blank(),
                          axis.text.x=element_blank(),
                          axis.ticks.x=element_blank(), 
                          axis.title.y=element_blank(),
                          axis.text.y=element_blank(),
                          axis.ticks.y=element_blank())

pdf("../figures/gene_gene_similarity_combine.pdf", width=8, height=7)
print(heat_all)
dev.off()

upper_tri <- get_upper_tri(cormat[501:520,501:520])
melted_cormat <- melt(upper_tri, na.rm = TRUE)

heat_20 <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.25, limit = c(0,1), space = "Lab", 
                       name="Similarity\nScore") +
  theme_minimal() + # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1))+
  coord_fixed() + theme(axis.title.x=element_blank(),
                        axis.title.y=element_blank())

pdf("../figures/gene_gene_similarity_combine_20.pdf", width=4, height=3)
print(heat_20)
dev.off()

# ---------------------------------------------------------------------------
# read in data and re-order genes based on annotations
# ---------------------------------------------------------------------------

dat = fread("../data/cts_all_but_Micro_Endo.txt.gz")
dim(dat)
dat[1:2,1:5]

table(genes$gene == dat$V1)

dim(cormat)
cormat[1:2,1:2]

table(rownames(cormat) %in% dat$V1)

mat1 = match(rownames(cormat), dat$V1)
dat1 = dat[mat1,]
dim(dat1)
dat1[1:2,1:5]

names(dat1)[1] = "gene_name"
fwrite(dat1, file="../data/cts_all_but_Micro_Endo_ordered_by_annotation.txt")
system("gzip ../data/cts_all_but_Micro_Endo_ordered_by_annotation.txt")

sessionInfo()
q(save = "no")

