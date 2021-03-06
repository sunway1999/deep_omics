
# Allen Brain MTG data

This dataset includes single nucleus RNA-seq data of 15,928 nuclei from Middle Temporal Gyrus (MTG) of human brains. It was generated using using SMART-Seq v4 Ultra Low Input RNA Kit, which is an improved version of SMART-seq2 protocol. More information can be found [here](https://celltypes.brain-map.org/rnaseq/human/mtg) and the associated publication: 

Hodge et al. Conserved cell types with divergent features in human versus mouse cortex. *Nature*. 2019 Sep;573(7772):61-8.


The whole dataset incluedes 10,708 excitatory neurons, 4,297 inhibitory neurons and 923 non-neuronal cells. While Hodge et al reported approximately 75 transcriptionally distinct cell types (subdivided into 45 inhibitory 
neuron types, 24 excitatory neuron types, and 6 non-neuronal types), here we focus on the larger categories: 

- Exc: excitatory neurons, or glutamatergic neurons
- Inh: inhibitory neurons, or GABAergic inhibitory interneurons
- Astro: astrocytes
- Endo: endothelial cells
- Micro: microglia
- Oligo: oligodendrocytes
- OPC: oligodendrocyte precursor cells

Two cell types microglia and endothelial cells have only a few nuclei and thus we skip them in some of the following analysis. 

## data

Processed data. The original data file `human_LGN_gene_expression_matrices_2018-06-14.zip` was downloaded from [here](http://celltypes.brain-map.org/api/v2/well_known_file_download/694416667). It was processed by an [R pipeline](https://github.com/Sun-lab/scRNAseq_pipelines/blob/master/MTG/human_MTG.Rmd) and the rendered html file can be viewed [here](https://htmlpreview.github.io/?https://github.com/Sun-lab/scRNAseq_pipelines/blob/master/MTG/human_MTG.html)

The raw data include counts from exons and introns separately. Since part of the reasons to prepare this pipeline is to prepare a reference for bulk RNA-seq data where gene expression are mostly measured on exonic regions. Here we only kept the exon counts.  

We filtered out 70 outlier nuclei from the 15,928 nuclei, which defined as being percent_reads_unique lower than 50%, and ended up with 15,858 nuclei. 


## python

Python code to construct autoencoder.

## R

R code to 

1. `step1_prepare_eData.R`: 
    - start with 4748 hvg's (highly varible genes).
    - remove genes with 0 count in more than 80% of nuclei, ended up with 1916 genes.
    - redefine cells of each cell type by taking intersection of cell type labels from Hodge et al. and clustering results. Here are the final cell/nuclei counts for each cell type. 

```
  Cell_Type nCells_All
1       Inh       4131
2       Exc      10434
3     Oligo        310
4       OPC        233
5     Astro        287
6     Micro         62
7      Endo          8
```


2. `step2_gene_simlarity.R`: generate gene-gene similarity.

3. `step3_save_clusters_to_csv.R`: save data in rds file to csv file. 

4. `step4_check_results.R`: check the auto-encoder results.


