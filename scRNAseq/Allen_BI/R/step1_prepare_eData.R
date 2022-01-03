
MTG_dir = "~/research/scRNAseq/data/Allen_BI/"
MTG_dir = paste0(MTG_dir, "human_MTG_gene_expression_matrices_2018-06-14")
setwd(MTG_dir)

library('org.Hs.eg.db')

# ------------------------------------------------------------------
# load clustering results 
# ------------------------------------------------------------------

sce      = readRDS("final_sce_hvg.rds")
clusters = readRDS("final_hvg_clust.rds")

dim(sce)
dim(colData(sce))
colData(sce)[1:2,1:5]

table(colData(sce)$cell_type, colData(sce)$class)

dim(clusters)
clusters[1:2,1:5]
names(clusters)

table(clusters$sample_name == colData(sce)$sample_name)
table(clusters$cell_type == colData(sce)$cell_type)

t1 = table(clusters$KM_15, clusters$cell_type)
t1

# based on manual examination of human_MTG.html, we choose to use the 
# clustering result of kmeans with 15 clusters. 
clusters$cluster_kmean = clusters$KM_15
clusts = apply(t1, 2, function(v){union(which.max(v), which(v > 200))})
clusts

# note that for some clusters, some cells belong to one cell type, 
# but other cells belong to another cell type. 
table(unlist(clusts))

# ------------------------------------------------------------------
# process geneInfo
# ------------------------------------------------------------------

geneInfo = as.data.frame(rowData(sce))
dim(geneInfo)
geneInfo[1:2,]
length(unique(geneInfo$gene))

columns(org.Hs.eg.db)
map1 = mapIds(org.Hs.eg.db, keys=as.character(geneInfo$entrez_id), 
              'ENSEMBL', 'ENTREZID')
length(map1)
map1[1:5]

geneInfo$ensembl_gene_id = as.character(map1)
table(names(map1) == geneInfo$entrez_id)

# ------------------------------------------------------------------
# remove genes with very low expression
# ------------------------------------------------------------------

setwd("~/research/GitHub/Vaeda")

# what are those genes with high expression and no ensembl gene id
w2check = geneInfo$pct_dropout_by_counts < 20 & is.na(geneInfo$ensembl_gene_id)
geneInfo[which(w2check),]

table(is.na(geneInfo$ensembl_gene_id))
wEn = which(!is.na(geneInfo$ensembl_gene_id))

ave_counts = geneInfo$total_counts/geneInfo$n_cells_by_counts
summary(ave_counts)
summary(geneInfo$pct_dropout_by_counts)

table(geneInfo$pct_dropout_by_counts < 95)
table(geneInfo$pct_dropout_by_counts < 90)
table(geneInfo$pct_dropout_by_counts < 80)

png("figures/gene_pct_dropout.png", width=4, heigh=4, units="in", res=400)
hist(geneInfo$pct_dropout_by_counts, main="", xlab="percentage of dropout")
dev.off()

png("figures/gene_filtering.png", width=8, heigh=4, units="in", res=400)
par(mfrow=c(1,2), bty="n")
plot(log10(ave_counts[-wEn]), geneInfo$pct_dropout_by_counts[-wEn], 
     pch=20, cex=0.3, col="grey", xlab="log10(mean counts)", 
     ylab="percent dropout", main="without Ensembl ID", 
     xlim=log10(range(ave_counts)))
abline(h=80)

plot(log10(ave_counts[wEn]), geneInfo$pct_dropout_by_counts[wEn], 
     pch=20, cex=0.3, col="red", xlab="log10(mean counts)",
     ylab="percent dropout", main="with Ensembl ID",
     xlim=log10(range(ave_counts)))
abline(h=80)
dev.off()

geneInfo[which(ave_counts > 10000),]

w2kp = geneInfo$pct_dropout_by_counts < 80
table(w2kp)
table(w2kp, !is.na(geneInfo$ensembl_gene_id))

sce = sce[which(w2kp),]
dim(sce)

geneInfo = geneInfo[which(w2kp),]
dim(geneInfo)
geneInfo[1:2,]

# ------------------------------------------------------------------
# collect counts for each cell type
# ------------------------------------------------------------------

write.table(geneInfo, file="data/gene_info.txt", sep="\t", 
            row.names=FALSE, col.names=TRUE, quote=FALSE)


celltypes = setdiff(unique(clusters$cell_type), "unknown")
celltypes

zeros  = rep(0,length(celltypes))
nCells = data.frame(Cell_Type=celltypes, nCells_All=zeros)

cells.all = NULL
ctype.all = NULL

for(ct1 in celltypes){
  ct.cond    = clusters$cell_type == ct1
  clust.cond = clusters$cluster_kmean %in% clusts[[ct1]]
  cells      = which(ct.cond & clust.cond)
  cells.all  = c(cells.all, cells)
  ctype.all  = c(ctype.all, rep(ct1, length(cells)))
  
  nCells[which(nCells$Cell_Type==ct1),2] = length(cells)

  ct.matrx = counts(sce)[,cells]
  write.table(ct.matrx, file=sprintf("data/cts_%s.txt", ct1), sep="\t", 
              row.names=TRUE, col.names=TRUE, quote=FALSE)
}

dim(nCells)
nCells

ct2kp = celltypes[1:5]
w2kp  = cells.all[which(ctype.all %in% ct2kp)]
ct.matrx = counts(sce)[,w2kp]
dim(ct.matrx)
write.table(ct.matrx, file="data/cts_all_but_Micro_Endo.txt", sep="\t", 
            row.names=TRUE, col.names=TRUE, quote=FALSE)

setwd("~/research/GitHub/Vaeda/data")
system("gzip -f *.txt")

sessionInfo()
q(save="no")


