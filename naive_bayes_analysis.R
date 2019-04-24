# ST 540: Applied Bayesian Analysis
# Authors: Andrew Emerson and Markel Sanz Ausin

# Main Goal: To explore the Naive Bayes classifier by comparing the "vanilla"
# version as a baseline to an approach that calculates the full conditional distributions
# of all covariates (removing conditional independence assumptions).

library(naivebayes)
library(rjags)
library(caret)

# Separating data into response and covariates.

# Letter Recognition (16 total covariates; discrete)
Y.letter.recognition <- letter.recognition[,1]
X.letter.recognition <- letter.recognition[,2:17]

# Iris (4 total covariates; continuous)
Y.iris <- iris[,5]
X.iris <- scale(iris[,1:4])

### Predictions Using Standard NB:

# Letter Recogntion:
Y.letter.recognition.train <- Y.letter.recognition[1:16000]
X.letter.recognition.train <- X.letter.recognition[1:16000,]
Y.letter.recognition.test <- Y.letter.recognition[16001:20000]
X.letter.recognition.test <- X.letter.recognition[16001:20000,]
nb.letter.recognition <- naive_bayes(X.letter.recognition.train, Y.letter.recognition.train)
preds.letter.recognition <- predict(nb.letter.recognition, X.letter.recognition.test)
letter.recognition.confusion.matrix <- confusionMatrix(preds.letter.recognition, Y.letter.recognition.test)

# Iris:
# TODO: Find different priors for the continuous covariates.



