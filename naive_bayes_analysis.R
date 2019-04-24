# ST 540: Applied Bayesian Analysis
# Authors: Andrew Emerson and Markel Sanz Ausin

# Main Goal: To explore the Naive Bayes classifier by comparing the "vanilla"
# version as a baseline to an approach that calculates the full conditional distributions
# of all covariates (removing conditional independence assumptions).

library(naivebayes)
library(rjags)

# Separating data into response and covariates.

# Letter Recognition (16 total covariates; discrete)
Y.letter.recognition <- letter.recognition[,1]
X.letter.recognition <- letter.recognition[,2:17]

# Iris (4 total covariates; continuous)
Y.iris <- iris[,5]
X.iris <- scale(iris[,1:4])
