#### intalling and loading libraries
library(BiocParallel)
register(MulticoreParam(8))

library(dplyr)
library(data.table)

library(Biobase)
library(org.Hs.eg.db)
library(OrganismDbi)


#### reading count matrix
library(data.table)
counts = fread("../data/GSE124326_count_matrix.txt.gz")
colnames(counts) <- sapply(colnames(counts), function(x){sub(".counts","",x)})
counts[,1:3] %>% head

### reading phenotypic data
pheno = fread("../data/GSE124326_pheno.txt")
pheno %>% head

#### Checking and mapping counts and phenotypic data
phenoNames = pheno$sample
pheno1 = pheno[,-c("sample")]
rownames(pheno1) <- phenoNames

countsNames <- sub("\\..*", "", counts$gene) # remove version number
counts1 = counts[,-c("gene")]
rownames(counts1) <- countsNames

#Remove the samples in counts that are missing in pheno 
missingNames <- setdiff(colnames(counts1), rownames(pheno1)) # elements of x not in y

counts1 <- counts1[, !c(..missingNames)]
rownames(counts1) <- countsNames

#check samples of counts match with those phenotypic information
all(colnames(counts1) == rownames(pheno1))

# convert to data frame
pheno1 <- as.data.frame(pheno1)
rownames(pheno1) <- phenoNames
counts1 <- as.data.frame(counts1)
rownames(counts1) <- countsNames
pheno1 %>% head

# Remove genes with low counts
keep <- (rowSums(counts1) >= 10)
counts1 <- counts1[keep,]
dim(counts1)

# combine BP1 and BP2 as BD
pheno1 <- pheno1 %>%
  mutate(condition = factor(ifelse(diagnosis %in% c("BP1", "BP2"), "BD", "Control"), levels = c("Control", "BD")))
rownames(pheno1) <- phenoNames
pheno1 %>% head
counts1 %>% head

# get the GO terms for the genes 
ensemble_keys <- unique(rownames(counts1))
tnf <- OrganismDbi::select(org.Hs.eg.db, keys=ensemble_keys, keytype="ENSEMBL", columns=c("ENTREZID", "GO"))
tnf <- tnf[tnf$ONTOLOGY == "BP", c("ENSEMBL", "GO")]
tnf %>% head

# save to csv
write.csv(tnf, "../data/annotations-gene-GO.csv", row.names = FALSE)
write.csv(t(counts1), "../data/counts1.csv", row.names = TRUE)
write.csv(pheno1, "../data/pheno1.csv", row.names = TRUE)

