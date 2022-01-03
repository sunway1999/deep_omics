
clusters = readRDS("../data/final_hvg_clust.rds")
dim(clusters)
names(clusters)

write.csv(clusters, file = "../data/final_hvg_custer.csv", row.names = FALSE)
