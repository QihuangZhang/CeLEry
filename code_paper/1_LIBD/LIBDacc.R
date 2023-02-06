## 0. Global Parameters and Packages

library(dplyr)
library(tidyr)
library(ggplot2)
## 1. Data  Processing

# classresults <- read.csv("output/LIBD/Multiple/data_gene_All_layerv2_1e-05_n2_probmat.csv", header = F)
# classresults <- read.csv("output/Hodge/layer_PreOrgv2_probmat.csv", header = F)
# classresults <- read.csv("output/Hodge/data_gene_All_layerv2_1e-05_n10_probmat.csv", header = F)

OverallAccSummary <- function (path) {
  classresults <- read.csv(path, header = F)
  
  classresults_new <-  classresults %>%
    mutate(Type = case_when( 
      V1 == V2 ~ "Same",
      abs(V1-V2) == 1 ~ "Neighbour",
      T ~ "Other"))
  
  summaries <- table(classresults_new$Type)
  
  exact_acc <- summaries["Same"]/sum(summaries)
  cat(exact_acc)
  Neighbor_acc <- exact_acc + summaries["Neighbour"]/sum(summaries)
  cat(Neighbor_acc)
  
  # for (i in 1:7) {
  #   data_curremt <-
  #   
  # }
  
  return( c(exact_acc, Neighbor_acc) )
}


ordinary507 <- OverallAccSummary("output/LIBD/Prediction151507/layer_PreOrg_probmat.csv")
ordinary676 <- OverallAccSummary("output/LIBD/Prediction151676/layer_PreOrg_probmat.csv")


aug507 <- OverallAccSummary("output/LIBD/Prediction151507/data_gen_layer_1e-05_n2_probmat.csv")
aug676 <- OverallAccSummary("output/LIBD/Prediction151676/data_gen_layer_1e-05_n2_probmat.csv")

# OverallAccSummary("output/LIBD/Prediction151507/data_gen_layer_1e-05_n2_probmat.csv")
# OverallAccSummary("output/LIBD/Prediction151676/data_gen_layer_1e-05_n2_probmat.csv")


multiple507 <- OverallAccSummary("output/LIBDmultiple/Prediction151507/layer_PreOrgv2_probmat.csv")
multiple676 <- OverallAccSummary("output/LIBDmultiple/Prediction151676/layer_PreOrgv2_probmat.csv")

multipleaug507 <- OverallAccSummary("output/LIBDmultiple/Prediction151507/data_gene_All_layerv2_1e-05_n2_probmat.csv")
multipleaug676 <- OverallAccSummary("output/LIBDmultiple/Prediction151676/data_gene_All_layerv2_1e-05_n2_probmat.csv")

Tangram507 <- OverallAccSummary("output/LIBD/Prediction151507/Tangram_decisionmat.csv")
Tangram676 <- OverallAccSummary("output/LIBD/Prediction151676/Tangram_decisionmat.csv")


spaOTsc507 <- OverallAccSummary("output/LIBD/Prediction151507/spaOTsc_decisionmat.csv")
spaOTsc676 <- OverallAccSummary("output/LIBD/Prediction151676/spaOTsc_decisionmat.csv")


novosparc507 <- OverallAccSummary("output/LIBD/Prediction151507/novosparc_decisionmat.csv")
novosparc676 <- OverallAccSummary("output/LIBD/Prediction151676/novosparc_decisionmat.csv")


accuracy_table <- rbind(ordinary507, ordinary676, aug507, aug676, multiple507, multiple676, multipleaug507, multipleaug676,
                    Tangram507, Tangram676, spaOTsc507, spaOTsc676, novosparc507, novosparc676)
colnames (accuracy_table) <- c("top1", "top2")

accuracy_table_long <- data.frame(accuracy_table) %>%
  add_rownames(var = "method") %>% 
  pivot_longer(cols = top1:top2, names_to = "type", values_to = "accuracy") %>%
  mutate( tissue = gsub('[A-Za-z]+', '', method))  %>%
  mutate( method = gsub('[0-9]+', '', method)) %>%
  mutate( Scenario = case_when(
    (!method %in% c("multiple", "multipleaug")) & (tissue == 676) ~ 1,
    (!method %in% c("multiple", "multipleaug")) & (tissue == 507) ~ 2,
    (method %in% c("multiple", "multipleaug")) & (tissue == 676) ~ 3,
    (method %in% c("multiple", "multipleaug")) & (tissue == 507) ~ 4
  )) %>%
  mutate( method =  factor (method, levels = unique(method), 
                            labels = c("CeLEry", "CeLEry(aug)", "CeLEry", "CeLEry(aug)", "Tangram", "spaOTsc", "novosparc") )) %>%
  mutate(type = factor(type, levels = unique(type), labels = c("top-1", "top-2"))) %>%
  mutate(Scenario = factor (Scenario, levels = c(1, 3, 2, 4), labels = paste("Scenario", c(1, 3, 2, 4)))) %>%
  data.frame()


pdf(file = "output/LIBD/plots/LIBD_barplot.pdf", width = 9, height = 9)

color_palatte <-c( "#CAE7B9", "#F3DE8A","#EB9486", "#7E7F9A", "#97A7B3")
strip_color <- "#0A1D37"

barplot <- ggplot(accuracy_table_long, aes(fill = method, x = type, y = accuracy)) + 
  geom_bar(stat = "identity", position="dodge") + 
  scale_fill_manual(values=color_palatte[c(3,2,1,4,5)]) + 
  facet_wrap(~Scenario)  +
  # scale_y_continuous(trans=scales::pseudo_log_trans(base = 10), breaks = c(0, 1, 10, 100, 500)) +
  theme_bw()  +
  theme(text=element_text(size=25, family="URWHelvetica"), axis.text = element_text(size = 25, family="URWHelvetica"), panel.spacing = unit(1, "lines") ) +
  theme(strip.background =element_rect(fill=strip_color,color=strip_color))+ # #535b44
  theme(strip.text = element_text(colour = 'white')) + # , axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  theme(panel.border = element_rect(colour = strip_color), legend.position = "bottom") +
  labs(fill = "Method", x = "Type", y = "Accuracy")

print(barplot)

dev.off()


pdf(file = "output/LIBD/plots/LIBD_barplot_legend.pdf", width = 12, height = 8)

print(barplot)

dev.off()


# ### Further exploration
# 
# OverallAccCeLEry <- function (path, probmat, truth) {
#   classresults <- read.csv(paste0(path,probmat), header = F)
#   truthresults <- read.csv(paste0(path,truth), header = F)
#   names(truthresults) <- "truth"
#   
#   classresults_new <-  data.frame(classresults, truth = truthresults) %>%
#     mutate(Type = case_when( 
#       V1 == truth ~ "Same",
#       abs(V1-truth) == 1 ~ "Neighbour",
#       T ~ "Other"))
#   
#   summaries <- table(classresults_new$Type)
#   
#   exact_acc <- summaries["Same"]/sum(summaries)
#   cat(exact_acc)
#   Neighbor_acc <- exact_acc + summaries["Neighbour"]/sum(summaries)
#   cat(Neighbor_acc)
#   
#   return( c(exact_acc, Neighbor_acc) )
# }
# 
# 
# OverallAccCeLEry(path = "output/LIBD/PredictionEmbd/", 
#                  probmat = "Emd_model_151673_151507_probmat.csv",
#                  truth = "Emd_model_151673_151507_truth.csv")
# 
# OverallAccCeLEry(path = "output/LIBD/PredictionEmbd/", 
#                  probmat = "Emd_model_151673_151676_probmat.csv",
#                  truth = "Emd_model_151673_151676_truth.csv")
