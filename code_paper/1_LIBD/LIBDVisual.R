### This file trying to visual the hodge results using the histology information

## 0. Global Parameters and Packages

library(dplyr)
library(ggplot2)
library(tidyr)
library(png)

outputdir <- "output/LIBD/plots/"



# Functions ---------------------------------------------------------------

Density_plot_data <- function(pred_result, Study){
  dataplot <- cbind(background, pred_result) 
  data_long <- gather(dataplot, column, prob, V3:V9, factor_key = TRUE)
  data_long_new <- data_long %>%
    mutate(TargetLayer = factor(column, labels = c("L1","L2","L3","L4","L5","L6","WM"), levels = paste0("V",3:9))) %>%
    mutate(study = Study)
  return(data_long_new)
}


labelvector <- c("L1","L2","L3","L4","L5","L6","WM")



Density_plot <- function(dataall, LayerSet = labelvector){
  png(file = paste0(outputdir,"LIBD_Density_plot.png"), height = 1600, width = 1400)
  Boxplot  <- ggplot(dataall %>%
                       filter (TargetLayer %in% LayerSet) %>%
                       mutate (TargetLayerfull = factor(TargetLayer, 
                                                       labels =  paste0("predicted to ",LayerSet),
                                                       levels = LayerSet)), 
                     aes(x = ycord, y = xcord) )  + 
    theme_bw()  + 
    geom_point(aes(fill = prob), shape = 21, color = "black",  size = 1.5, stroke = 0.3) +
    annotation_raster(Histology, ymin = 0, ymax= 1, xmin = 0, xmax = 1) +
    # scale_y_reverse() +
    facet_grid(TargetLayerfull~study, switch = "y") +
    # scale_fill_gradient(low = "#08121b", high = "#9fd3fa") +
    scale_fill_gradient2(low = "#08121b", mid = "#56b1f7", high = "#EB9486",midpoint = 0.5) +
    theme(text=element_text(size=25, family="URWHelvetica"), axis.text = element_text(size = 30, family="URWHelvetica"),
          panel.spacing = unit(1, "lines")) +
    theme(strip.background =element_rect(fill="#17202A",color="#17202A"))+ 
    theme(strip.text = element_text(colour = 'white'),axis.text=element_blank()) + #, strip.position = "left"
    theme(panel.border = element_rect(colour = "#17202A"), legend.position="none")  + #
    labs(x = NULL, y = NULL, fill = "Probability")
  print(Boxplot)
  dev.off()
  
  png(file = paste0(outputdir,"LIBD_Density_plot_legend.png"), height = 1600, width = 1400)
  Boxplot2 <- Boxplot +
    theme(legend.position="right")
  
  print(Boxplot2)
  dev.off()
}

# 1. Prepare the LIBD backgound data --------------------------------------

## Using data of 507 as the background

tissue_pos <- read.csv("data/LIBD/visualization_151507.csv")
# Histology <- readPNG("data/LIBD/151507_tissue_lowres_image.png")
Histology <- readPNG("data/LIBD/RegionReference_bw2.png")
# Histology_tsp <- matrix(rgb(Histology[,,1],Histology[,,2],Histology[,,3], 0.7), nrow=dim(Histology)[1])

Histology_maxx <- max(tissue_pos["x2"])+1
Histology_minx <- min(tissue_pos["x2"])
Histology_maxy <- max(tissue_pos["x3"])+1
Histology_miny <- min(tissue_pos["x3"])


background <- tissue_pos %>%
  mutate (xcord = 0.92-x2/(1.2*Histology_maxx+Histology_minx-5)) %>%
  mutate (ycord = 0.12+x3/(1.22*Histology_maxy+Histology_miny))

# 2. Apply the Hodge results on to the LIBD background --------------------

## Load the results information

pred_CeLEry <- read.csv("output/LIBD/Prediction151507/layer_PreOrg_probmat.csv", header = F)
data_CeLEry <- Density_plot_data(pred_CeLEry, "CeLEry")

pred_CeLEryn2 <- read.csv("output/LIBD/Prediction151507/data_gen_layer_1e-05_n2_probmat.csv", header = F)
data_CeLEryn2 <- Density_plot_data(pred_CeLEryn2, "CeLEry (Augmentation)")

pred_Tangram <- read.csv("output/LIBD/Prediction151507/Tangram_probmat_151507.csv", header = F)
pred_Tangram_full <- cbind(0,0,pred_Tangram)
names(pred_Tangram_full) <- paste0("V",1:9)
data_Tangram <- Density_plot_data(pred_Tangram_full, "Tangram")

pred_Multiple <- read.csv("output/LIBDmultiple/Prediction151507/layer_PreOrgv2_probmat.csv", header = F)
data_Multiple <- Density_plot_data(pred_Multiple, "CeLEry (Multiple)")



pred_spaOTsc <- read.csv("output/LIBD/Prediction151507/spaOTsc_probmat.csv", header = F)
pred_spaOTsc_prop <- pred_spaOTsc/rowSums(pred_spaOTsc)
pred_spaOTsc2 <- read.csv("output/LIBD/Prediction151507/spaOTsc_decisionmat.csv", header = F)
pred_spaOTsc_full <- cbind(pred_spaOTsc2,pred_spaOTsc_prop)
names(pred_spaOTsc_full) <- paste0("V",1:9)
data_spaOTsc <- Density_plot_data(pred_spaOTsc_full, "spaOTsc")


pred_novosparc <- read.csv("output/LIBD/Prediction151507/novosparc_probmat.csv", header = F)
pred_novosparc_prop <- pred_novosparc/rowSums(pred_novosparc)
pred_novosparc2 <- read.csv("output/LIBD/Prediction151507/novosparc_decisionmat.csv", header = F)
pred_novosparc_full <- cbind(pred_novosparc2,pred_novosparc_prop)
names(pred_novosparc_full) <- paste0("V",1:9)
data_novosparc <- Density_plot_data(pred_novosparc_full, "novosparc")


data_all <- rbind(data_CeLEry, data_CeLEryn2, data_Tangram, data_Multiple, data_spaOTsc, data_novosparc)

Density_plot(data_all)

# 3. PredictionHodge: Other methods (Discarded) --------------------
# ## Tangram
# pred_Tangram <- read.csv("output/Hodge/PredictionHodge/Tangram_probmat.csv", header = F)
# 
# for (i in 1:6){
#   Density_plot(pred_Tangram, "Tangram", i)
# }
# 
# ## ClusterBased
# pred_ClusterBased <- read.csv("output/Hodge/PredictionHodge/ClusterBased_probmat.csv", header = F)
# 
# for (i in 1:6){
#   Density_plot(pred_ClusterBased, "ClusterBased", i)
# }

