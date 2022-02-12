
library(maftools)
library(ggplot2)
require(plyr)
library(cowplot)
library(dplyr)
library(ggrepel)
library(ggsci)
library(gfplot)

library(extrafont)
loadfonts(device = "win")


################################################# data processing

## ampca

ampca.maf = read.csv('src/ampca_bcm_2016/data_mutations_extended.txt', sep = "\t", stringsAsFactors = FALSE) 
ampca.clin = read.csv('src/ampca_bcm_2016/data_clinical_patient.txt', sep = "\t", stringsAsFactors = FALSE) 

ampca.clin.a = read.csv('src/Ampullary Cancer-Baylor College of Medicine, Cell Reports 2016-Pancreatobiliary.csv',stringsAsFactors = FALSE) 

inds <- intersect(ampca.clin.a$Sample.ID,ampca.clin$Tumor_Sample_Barcode)

ampca.maf.selcted  <- ampca.maf[which(ampca.maf$Tumor_Sample_Barcode%in%inds),]
ampca.clin.selcted  <- ampca.clin[which(ampca.clin$Tumor_Sample_Barcode%in%inds),]

ampca.selcted = read.maf(maf = ampca.maf.selcted, clinicalData = ampca.clin.selcted)


## pdac

pdac.maf = read.csv('src/paad_tcga_pan_can_atlas_2018/data_mutations_extended.txt', sep = "\t", stringsAsFactors = FALSE) 
pdac.clin = read.csv('src/paad_tcga_pan_can_atlas_2018/data_clinical_patient.txt', sep = "\t", stringsAsFactors = FALSE) 

pdac = read.maf(maf = pdac.maf, clinicalData = pdac.clin)


## chol

chol.maf = read.csv('src/chol_icgc_2017/data_mutations_extended.txt', sep = "\t", stringsAsFactors = FALSE) 
chol.clin = read.csv('src/chol_icgc_2017/data_clinical_patient.txt', sep = "\t", stringsAsFactors = FALSE) 

chol.clin.a = read.csv('src/chol-ICGC Cancer Discov 2017-Fluke-Neg.csv',stringsAsFactors = FALSE) 

inds <- intersect(chol.clin.a$Sample.ID,chol.clin$Tumor_Sample_Barcode)

chol.maf.selcted   <- chol.maf[which(chol.maf$Tumor_Sample_Barcode%in%inds),]
chol.clin.selcted  <- chol.clin[which(chol.clin$Tumor_Sample_Barcode%in%inds),]

chol.selcted = read.maf(maf = chol.maf.selcted, clinicalData = chol.clin.selcted)


save(ampca.selcted,pdac,chol.selcted,ampca_genes_selected,
     ampca.vs.pdac,ampca.vs.chol,
     file = "rslt/V1/mutation.RData")





################################################# Figure 

load("rslt/V1/mutation.RData")

#### 

ampca_summary <- getGeneSummary(ampca.selcted)
ampca_genes_selected <- ampca_summary$Hugo_Symbol[1:50]

## A

vc_cols = c("#E72223","#CD6090","#F4A35D","#EEC0C3",
            "#008B8B","#52C4F2","#000000","#727272")
names(vc_cols) = c('Missense_Mutation','Frame_Shift_Del','Nonsense_Mutation','Frame_Shift_Ins',
                   'Splice_Site','In_Frame_Del','In_Frame_Ins','Multi_Hit')

oncoplot(
  maf = ampca.selcted,
  genes = ampca_genes_selected,
  colors = vc_cols,
  bgCol = "#D5D5D5")

## B

oncoplot(
  maf = pdac,
  genes = ampca_genes_selected,
  colors = vc_cols,
  bgCol = "#D5D5D5")

## C

oncoplot(
  maf = chol.selcted,
  genes = ampca_genes_selected,
  colors = vc_cols,
  bgCol = "#D5D5D5")


## D E

coBarplot(genes = ampca_genes_selected,m1 = pdac, m2 = ampca.selcted , m1Name = "pdac", m2Name = "ampca",colors = vc_cols)

coBarplot(genes = ampca_genes_selected,m1 = ampca.selcted, m2 = chol.selcted, m1Name = "ampca", m2Name = "chol",colors = vc_cols)



#### comparison

ampca.vs.pdac <- mafCompare(m1 = ampca.selcted, m2 = pdac, m1Name = 'ampca', m2Name = 'pdac', minMut = 5)
ampca.vs.pdac <- ampca.vs.pdac$results

ampca.vs.chol <- mafCompare(m1 = ampca.selcted, m2 = chol.selcted, m1Name = 'ampca', m2Name = 'chol', minMut = 5)
ampca.vs.chol <- ampca.vs.chol$results





