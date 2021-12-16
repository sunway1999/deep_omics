
library(Matrix)

# ---------------------------------------------------------------------------
# read in clinical information per patient
# ---------------------------------------------------------------------------

patient = read.table("../msk_impact_2017/data_clinical_patient.txt", 
                         header=TRUE, sep="\t", as.is=TRUE)
dim(patient)
head(patient)

table(patient$SEX, useNA="ifany")
table(patient$SMOKING_HISTORY, useNA="ifany")
table(patient$VITAL_STATUS, patient$OS_STATUS, useNA="ifany")
summary(patient$OS_MONTHS)

length(unique(patient$PATIENT_ID))

# ---------------------------------------------------------------------------
# read in clinical information per sample
# ---------------------------------------------------------------------------

samples = read.table("../msk_impact_2017/data_clinical_sample.txt", 
                        header=TRUE, sep="\t", as.is=TRUE, quote="",
                        comment.char = "", skip=4)
dim(samples)
head(samples)
length(unique(samples$PATIENT_ID))
length(unique(samples$SAMPLE_ID))

table(samples$SAMPLE_COLLECTION_SOURCE, useNA="ifany")
table(samples$SPECIMEN_PRESERVATION_TYPE, useNA="ifany")
table(samples$SPECIMEN_TYPE, useNA="ifany")

summary(samples$DNA_INPUT)
summary(samples$SAMPLE_COVERAGE)

pdf("../figures/DNA_input_vs_coverage.pdf", width=9, height=3)
par(mfrow=c(1,3), mar=c(5,4,1,1), bty="n")
hist(samples$DNA_INPUT, xlab="DNA input", main="")
hist(samples$SAMPLE_COVERAGE, xlab="Sample coverage", main="")
DNAinput.cat = cut(samples$DNA_INPUT, breaks=c(50, 100, 200, 250), 
                   include.lowest=TRUE)
boxplot(samples$SAMPLE_COVERAGE ~ DNAinput.cat, 
        xlab="DNA input", ylab="Sample coverage")
dev.off()

table(samples$DNA_INPUT > 100, samples$SAMPLE_COVERAGE > 100)
table(samples$DNA_INPUT > 100, samples$SAMPLE_COVERAGE > 200)

samples = samples[samples$DNA_INPUT > 100 & samples$SAMPLE_COVERAGE > 200,]
dim(samples)
samples[1:2,]


table(samples$MATCHED_STATUS)
summary(samples$TUMOR_PURITY)
table(samples$SAMPLE_TYPE)

table(samples$SAMPLE_TYPE, samples$SPECIMEN_PRESERVATION_TYPE, 
      useNA="ifany")

table(samples$SAMPLE_TYPE, samples$SPECIMEN_TYPE, useNA="ifany")


# ---------------------------------------------------------------------------
# read in mutation data
# ---------------------------------------------------------------------------

mut.extended = read.table("../msk_impact_2017/data_mutations_extended.txt", 
                          header=TRUE, sep="\t", as.is=TRUE, quote="")
dim(mut.extended)
head(mut.extended)
length(unique(mut.extended$Tumor_Sample_Barcode))
length(unique(mut.extended$Hugo_Symbol))

table(mut.extended$NCBI_Build, useNA="ifany")
sort(table(mut.extended$Consequence, useNA="ifany"), decreasing=T)[1:20]
table(mut.extended$Variant_Classification, useNA="ifany")

total_count = mut.extended$t_ref_count + mut.extended$t_alt_count
summary(mut.extended$t_alt_count/total_count)
ww1 = which(mut.extended$t_alt_count/total_count > 1)
mut.extended[ww1,]

# ---------------------------------------------------------------------------
# read in MSKCC mutation data
# ---------------------------------------------------------------------------

mut.mskcc = read.table("../msk_impact_2017/data_mutations_mskcc.txt", 
                       header=TRUE, sep="\t", as.is=TRUE, quote="")
dim(mut.mskcc)
head(mut.mskcc)
length(unique(mut.mskcc$Tumor_Sample_Barcode))
length(unique(mut.mskcc$Hugo_Symbol))
table(mut.mskcc$Variant_Classification, useNA="ifany")

mskcc = samples$SAMPLE_ID %in% mut.mskcc$Tumor_Sample_Barcode
table(samples$SAMPLE_COLLECTION_SOURCE, mskcc)

table(mut.mskcc$Tumor_Sample_Barcode %in% mut.extended$Tumor_Sample_Barcode)

# ---------------------------------------------------------------------------
# it looks like MSKCC mutation data is a subset of extended mutation data
# though it is not clear what is the difference. 
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prepare a data matrix of all the mutations and all the samples
# ---------------------------------------------------------------------------

mut2use = mut.extended[which(mut.extended$Tumor_Sample_Barcode %in% samples$SAMPLE_ID),]
dim(mut2use)

mat1 = match(mut2use$Tumor_Sample_Barcode, samples$SAMPLE_ID)
table(is.na(mat1))
mut2use$PATIENT_ID = samples$PATIENT_ID[mat1]
sam2use = samples[mat1,]

# ---------------------------------------------------------------------------
# one patient may have multiple samples, chose the one with largest 
# number of mutations
# ---------------------------------------------------------------------------

tb0 = tapply(sam2use$SAMPLE_ID, sam2use$PATIENT_ID, table)
n.per.patient = sapply(tb0, length)
table(n.per.patient)

tb1 = lapply(tb0, sort, decreasing=TRUE)
length(tb1)

tb0[which(n.per.patient==2)[1:5]]
tb1[which(n.per.patient==2)[1:5]]

sampleID2use = sapply(tb1, function(x){names(x)[1]})
length(sampleID2use)
sampleID2use[1:5]

dim(mut2use)
mut2use = mut2use[which(mut2use$Tumor_Sample_Barcode %in% sampleID2use),]
dim(mut2use)

length(unique(mut2use$Tumor_Sample_Barcode))
length(unique(mut2use$PATIENT_ID))

table(mut2use$PATIENT_ID == substr(mut2use$Tumor_Sample_Barcode, 1, 9))

mut2use = unique(mut2use[,c("Hugo_Symbol", "Tumor_Sample_Barcode")])
dim(mut2use)
head(mut2use)

# ---------------------------------------------------------------------------
# read in genes of the two platforms
# ---------------------------------------------------------------------------

genes341 = scan(file="../msk_impact_2017/data_gene_panel_impact341.txt", 
                skip=3, what=character())
genes341[1:5]
genes341 = genes341[-1]

genes410 = scan(file="../msk_impact_2017/data_gene_panel_impact410.txt", 
                skip=3, what=character())
genes410[1:5]
genes410 = genes410[-1]

table(genes341 %in% genes410)
genes.add = setdiff(genes410, genes341)
table(mut2use$Hugo_Symbol %in% genes341)
table(mut2use$Hugo_Symbol %in% genes410)

t1 = table(mut2use$Hugo_Symbol[mut2use$Hugo_Symbol %in% genes341])
summary(as.numeric(t1))

t2 = table(mut2use$Hugo_Symbol[mut2use$Hugo_Symbol %in% genes.add])
summary(as.numeric(t2))
sort(t2, decreasing=TRUE)[1:20]

# ---------------------------------------------------------------------------
# only use the genes in the 341 platform
# ---------------------------------------------------------------------------

mut2use = mut2use[which(mut2use$Hugo_Symbol %in% genes341),]
dim(mut2use)

genes    = sort(unique(mut2use$Hugo_Symbol))
patients = sort(unique(mut2use$Tumor_Sample_Barcode))
length(genes)
length(patients)

gene.index = match(mut2use$Hugo_Symbol, genes)
pati.index = match(mut2use$Tumor_Sample_Barcode, patients)

mut.matrix = sparseMatrix(i=gene.index, j=pati.index, 
                          x=rep(1, nrow(mut2use)), 
                          dimnames=list(genes, patients))
dim(mut.matrix)
mut.matrix[1:2,1:4]

mut.matrix1 = as(mut.matrix, "matrix")
dim(mut.matrix1)
mut.matrix1[1:2,1:4]

mload = colSums(mut.matrix1)
table(mload >= 3)
table(mload >= 5)
table(mload >= 6)
table(mload >= 8)
table(mload >= 10)

mut.df = data.frame(geneName=rownames(mut.matrix1), mut.matrix1)
dim(mut.df)
mut.df[1:2,1:4]

write.table(mut.df, file='../data/mut_matrix_340_x_9112.txt', sep="\t", 
            row.names=FALSE, col.names=TRUE, quote=FALSE)

sessionInfo()
q(save="no")

