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

nb.iris <- naive_bayes(X.iris.train, Y.iris.train, prior=)
preds.iris <- predict(nb.iris, X.iris.test)
iris.confusion.matrix <- confusionMatrix(preds.iris, Y.iris.test)



##################################################################################
# JAGS
# IRIS
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
  #mua ~ dnorm(0, 0.001)
  #mub ~ dnorm(0, 0.001)
  taua ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
}")

model <- jags.model(model_string, data=data, n.chains=n.chains, quiet=T)
update(model, burn)
samples <- coda.samples(model, variable.names=params, thin=thin, n.iter=n.iter)
plot(samples)
summary(samples)
gelman.diag(samples)
effectiveSize(samples)

res <- c(summary(samples)$quantiles[1:38,3] == as.numeric(Y.iris.test))
print(paste("Accuracy:", length(subset(res, res==T)) / length(res)))


# HAND-WRITTEN LETTERS

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
    for (j in 1:26) {
      logit(pi[i,j]) <- alpha[j] + inprod(beta[j,], X[i,])
    }
  }
  # Prediction
  for (i in 1:n_pred) {
    Y_pred[i] ~ dcat(pi_pred[i,])
    for (j in 1:26) {
      logit(pi_pred[i,j]) <- alpha[j] + inprod(beta[j,], X_pred[i,])
    }
  }
  # Priors
  for (j in 1:26) {
    alpha[j] ~ ddexp(0, taua)
    for (k in 1:p) {
      beta[j,k] ~ ddexp(0, taub)
    }
  }
  #mua ~ dnorm(0, 0.001)
  #mub ~ dnorm(0, 0.001)
  taua ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
  }")

model <- jags.model(model_string, data=data, n.chains=n.chains)
update(model, burn)
samples <- coda.samples(model, variable.names=params, thin=thin, n.iter=n.iter)
plot(samples)
summary(samples)
gelman.diag(samples)
effectiveSize(samples)

result <- samples[[1]]
result[1:38]
summary(samples)$quantiles[1:38,3] == as.numeric(Y.iris.test)

