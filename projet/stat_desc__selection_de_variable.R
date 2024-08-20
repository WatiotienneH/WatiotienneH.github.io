library(gtsummary)
library(dplyr)
library(survival)
library("survminer")
library(fitdistrplus)
library(gtsummary)
library(broom)
library(survminer)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(GGally)
library(summarytools)
data=read.csv2("C:/Users/hrywa/OneDrive/Bureau/Méthodes_d_apprentissage/projet/Data2.csv",sep = ",")

view(dfSummary(data))

data <- data %>%
  mutate(Label = factor(Label))


tbl <- tbl_summary(
  data, 
  by = Label, # Remplacez 'Label' par le nom de votre variable de groupe
  type = all_continuous() ~ "continuous2", # Pour appliquer le test t aux variables continues
) %>% add_p(test = all_continuous() ~ t.test) # Ajouter le test t pour les variables continues

tbl


# Identifier les lignes dupliquées dans l'ensemble de données
doublons <- data[duplicated(data), ]

# Compter le nombre de doublons par valeur unique de Label
doublons_count <- doublons %>%
  group_by(Label) %>%
  summarise(Nombre_de_doublons = n())

# Afficher le tableau récapitulatif des doublons
print(doublons_count)


