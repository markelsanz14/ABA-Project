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

train.set.percentage <- 0.75
set.seed(5) # Setting seed for reproducible results.

### Predictions Using Standard NB as Baseline:

# Letter Recogntion:
letter.recognition.sample.size <- floor(train.set.percentage * length(Y.letter.recognition))
letter.recognition.train.ind <- sample(seq_len(length(Y.letter.recognition)), size = letter.recognition.sample.size)

Y.letter.recognition.train <- Y.letter.recognition[letter.recognition.train.ind]
X.letter.recognition.train <- X.letter.recognition[letter.recognition.train.ind,]
Y.letter.recognition.test <- Y.letter.recognition[-letter.recognition.train.ind]
X.letter.recognition.test <- X.letter.recognition[-letter.recognition.train.ind,]

nb.letter.recognition <- naive_bayes(X.letter.recognition.train, Y.letter.recognition.train)
preds.letter.recognition <- predict(nb.letter.recognition, X.letter.recognition.test)
letter.recognition.confusion.matrix <- confusionMatrix(preds.letter.recognition, Y.letter.recognition.test)

# Iris:
iris.sample.size <- floor(train.set.percentage * length(Y.iris))
iris.train.ind <- sample(seq_len(length(Y.iris)), size = iris.sample.size)

Y.iris.train <- Y.iris[iris.train.ind]
X.iris.train <- X.iris[iris.train.ind,]
Y.iris.test <- Y.iris[-iris.train.ind]
X.iris.test <- X.iris[-iris.train.ind,]

nb.iris <- naive_bayes(X.iris.train, Y.iris.train)
preds.iris <- predict(nb.iris, X.iris.test)
iris.confusion.matrix <- confusionMatrix(preds.iris, Y.iris.test)


