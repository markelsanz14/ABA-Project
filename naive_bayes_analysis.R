# ST 540: Applied Bayesian Analysis
# Authors: Andrew Emerson and Markel Sanz Ausin

# Main Goal: To explore the Naive Bayes classifier by comparing the "vanilla"
# version as a baseline to an approach that calculates the full conditional distributions
# of all covariates (removing conditional independence assumptions).

library(naivebayes)
library(rjags)
library(caret)

#######################################################################################
# IRIS (4 total covariates; continuous)
#######################################################################################

set.seed(5) # Setting seed for reproducible results.
iris <- iris[sample(nrow(iris)),] # Randomize dataset
Y.iris <- iris[,5]
X.iris <- scale(iris[,1:4])

accuracies.iris.nb <- rep(0, 10)
accuracies.iris.jags <- rep(0, 10)
for (cv in 1:10) {
  val.start.ind <- (cv-1)*15 + 1
  Y.iris.test <- Y.iris[val.start.ind:(val.start.ind+14)]
  X.iris.test <- X.iris[val.start.ind:(val.start.ind+14), ]
  Y.iris.train <- Y.iris[-(val.start.ind:(val.start.ind+14))]
  X.iris.train <- X.iris[-(val.start.ind:(val.start.ind+14)),]
  
  ##################################################################################
  # NB
  nb.iris <- naive_bayes(X.iris.train, Y.iris.train)
  preds.iris <- predict(nb.iris, X.iris.test)
  iris.confusion.matrix <- confusionMatrix(preds.iris, Y.iris.test)
  accuracies.iris.nb[cv] <- iris.confusion.matrix$overall[1]
  
  ##################################################################################
  # JAGS
  n <- length(Y.iris.train)
  p <- ncol(X.iris.train)
  data <- list(Y=as.numeric(Y.iris.train), X=X.iris.train, X_pred=X.iris.test, n=n, p=p, n_pred=length(Y.iris.test))
  params <- c("beta", "alpha", "Y_pred")
  burn <- 10000
  n.iter <- 20000
  thin <- 10
  n.chains <- 2
  
  model_string = textConnection("model{
    # Likelihood
    for (i in 1:n) {
      Y[i] ~ dcat(pi[i,])
      logit(pi[i,1]) <- alpha[1] + inprod(beta[1,], X[i,])
      logit(pi[i,2]) <- alpha[2] + inprod(beta[2,], X[i,])
      logit(pi[i,3]) <- alpha[3] + inprod(beta[3,], X[i,])
    }
    # Prediction
    for (i in 1:n_pred) {
      Y_pred[i] ~ dcat(pi_pred[i,])
      logit(pi_pred[i,1]) <- alpha[1] + inprod(beta[1,], X_pred[i,])
      logit(pi_pred[i,2]) <- alpha[2] + inprod(beta[2,], X_pred[i,])
      logit(pi_pred[i,3]) <- alpha[3] + inprod(beta[3,], X_pred[i,])
    }
    # Priors
    for (j in 1:3) {
      alpha[j] ~ ddexp(0, taua)
      for (k in 1:p) {
        beta[j,k] ~ ddexp(0, taub)
      }
    }
    taua ~ dgamma(0.1, 0.1)
    taub ~ dgamma(0.1, 0.1)
}")
  
  model <- jags.model(model_string, data=data, n.chains=n.chains, quiet=T)
  update(model, burn)
  samples <- coda.samples(model, variable.names=params, thin=thin, n.iter=n.iter)
  
  res <- c(summary(samples)$quantiles[1:15,3] == as.numeric(Y.iris.test))
  accuracies.iris.jags[cv] <- length(subset(res, res==T)) / length(res)
  print(paste("CV fold", cv))
  print(paste("ACCURACY for NB:", accuracies.iris.nb[cv]))
  print(paste("ACCURACY for Jags:", accuracies.iris.jags[cv]))
}



#######################################################################################
# LETTER RECOGNITION (16 total covariates; discrete)
#######################################################################################

# Reducing size of letter recognition dataset, dropping unused levels, resetting index
letter.recognition <- letter.recognition[which (letter.recognition$V1 == "A" |
                                                letter.recognition$V1 == "B" |
                                                letter.recognition$V1 == "C"),]
letter.recognition$V1 <- droplevels(letter.recognition$V1)
letter.recognition <- letter.recognition[sample(nrow(letter.recognition)), ] # Randomize
letter.recognition <- letter.recognition[1:1000, ]
rownames(letter.recognition) <- NULL

Y.letter.recognition <- letter.recognition[,1]
X.letter.recognition <- letter.recognition[,2:17]

set.seed(5) # Setting seed for reproducible results.

### Predictions Using Standard NB as Baseline:
accuracies.letter.nb <- rep(0, 10)
accuracies.letter.jags <- rep(0, 10)
for (cv in 1:10) {
  val.start.ind <- (cv-1)*100 + 1
  Y.letter.recognition.test <- Y.letter.recognition[val.start.ind:(val.start.ind+99)]
  X.letter.recognition.test <- X.letter.recognition[val.start.ind:(val.start.ind+99), ]
  Y.letter.recognition.train <- Y.letter.recognition[-(val.start.ind:(val.start.ind+99))]
  X.letter.recognition.train <- X.letter.recognition[-(val.start.ind:(val.start.ind+99)),]
  
  ##################################################################################
  # NB
  nb.letter.recognition <- naive_bayes(X.letter.recognition.train, Y.letter.recognition.train)
  preds.letter.recognition <- predict(nb.letter.recognition, X.letter.recognition.test)
  letter.recognition.confusion.matrix <- confusionMatrix(preds.letter.recognition, Y.letter.recognition.test)
  accuracies.letter.nb[cv] <- letter.recognition.confusion.matrix$overall[1]
  
  ##################################################################################
  # JAGS
  n <- length(Y.letter.recognition.train)
  p <- ncol(X.letter.recognition.train)
  data <- list(Y=as.numeric(Y.letter.recognition.train), X=X.letter.recognition.train, 
               X_pred=X.letter.recognition.test, n=n, p=p, n_pred=length(Y.letter.recognition.test))
  params <- c("beta", "alpha", "Y_pred")
  burn <- 1000
  n.iter <- 2000
  thin <- 10
  n.chains <- 2
  
  model_string = textConnection("model{
    # Likelihood
    for (i in 1:n) {
      Y[i] ~ dcat(pi[i,])
      for (j in 1:3) {
        logit(pi[i,j]) <- alpha[j] + inprod(beta[j,], X[i,])
      }
    }
    # Prediction
    for (i in 1:n_pred) {
      Y_pred[i] ~ dcat(pi_pred[i,])
      for (j in 1:3) {
        logit(pi_pred[i,j]) <- alpha[j] + inprod(beta[j,], X_pred[i,])
      }
    }
    # Priors
    for (j in 1:3) {
      alpha[j] ~ ddexp(0, taua)
      for (k in 1:p) {
        beta[j,k] ~ ddexp(0, taub)
      }
    }
    taua ~ dgamma(0.1, 0.1)
    taub ~ dgamma(0.1, 0.1)
    }")
  
  model <- jags.model(model_string, data=data, n.chains=n.chains, quiet=T)
  update(model, burn)
  samples <- coda.samples(model, variable.names=params, thin=thin, n.iter=n.iter)
  
  res <- c(summary(samples)$quantiles[1:100,3] == as.numeric(Y.letter.recognition.test))
  accuracies.letter.jags[cv] <- length(subset(res, res==T)) / length(res)
  print(paste("Letter-Recognition: CV fold", cv))
  print(paste("ACCURACY for NB:", accuracies.letter.nb[cv]))
  print(paste("ACCURACY for Jags:", accuracies.letter.jags[cv]))
}
