# 🎯 Model Based - Clustering

Ce projet consiste à implémenter un package R pour la modélisation de mélanges gaussiens avec détection d’outliers.

## 📂 Structure du projet

- `mixturemodel/` — scripts R
- `/` — fichier ".tar.gz" à installer

## ⚙️ Installation

install.packages("chemin/vers/mixturemodel_0.1.0.tar.gz", repos = NULL, type = "source")

## 🔎 Exemple d’utilisation

library(mixturemodel)

model<-MixtureModel(X, K=2, initialization_steps = 20)
model<-fit(model)
tail(model$params)
