# Matthew Sudmann-Day
# Barcelona GSE Data Science
# Ridge and Lasso regressions.

# A function to perform a linear regression.
# Returns a vector of estimators for the features in X.
lin.reg <- function(y, X) {
  solve(t(X) %*% X, t(X) %*% y)
}

# A function to perform a ridge regression.
# Returns a vector of estimators for the features in X.
ridge.reg <- function(y, X, lambda) {
  solve(t(X) %*% X + diag(lambda, ncol(X)) , t(X) %*% y)
}

# A function to do a regression using the lasso shooting algorithm.
# Returns a vector of estimators for the features in X.
lasso.reg <- function(y, X, lambda) {
  # Use the result of a linear regression as our starting point.
  # Also assign beta to this to make sure beta has the right dimension.
  beta <- beta.prev <- lin.reg(y, X)

  # Iterate up to 100 times attempting to reduce our deviance.
  for (iter in 1:100) { 
    for (col in 1:ncol(X)) {
      # Variance and covariance calculations.
      var <- sum(X[,col] ^ 2)
      cov <- sum((y-X[,-col] %*% beta[-col]) * X[,col])
      cv <- cov/var
      
      # Calculate new values for beta.
      beta[col] <- max(0, abs(cv) - lambda/(2 * var)) * sign(cv)
    }

    # If only a neglible improvement was made, return our latest beta.
    if (sum((beta-beta.prev) ^ 2) < 1e-6) return(beta)
    beta.prev <- beta
  }
  return(beta)
}

# Perform 5-fold cross-validation to obtain an RSS value that runs a
# regression (lasso or ridge) on 80% of the data and applies it to the
# remaining 20%.
cross.validation <- function(y, X, lambda, pen.reg) {
  r <- nrow(X)/5
  RSSes <- c()
  
  for (slice in 0:4) {
    top <- slice * r + 1
    bottom <- top + r - 1
    small.X <- X[top:bottom,]
    small.y <- y[top:bottom]
    big.X <- X[-top:-bottom,]
    big.y <- y[-top:-bottom]
    
    # Get coefficients from the appropriate regression function.
    beta <- pen.reg(big.y, big.X, lambda)

    # Calculate RSS for this iteration and add it to a vector of RSS's.
    RSSes <- c(RSSes, sum((small.y - (small.X %*% beta)) ^ 2))
  }
  
  # Return the average RSS value.
  return(mean(RSSes))
}


set.seed(12345)
b = c(4,5,6,7,8,9,10)
X = matrix(0,100,7)
y = rnorm(100,0,0.2)*sum(b)*rnorm(1,1,0.1)
for (i in 1:100)
{
  r<-rnorm(7,0,0.2)
  X[i,]<-r*b+b
}
lin.reg(y, X)
lasso.reg(y, X, 0.1)



set.seed(12345)
b = c(4,5,6,7,8,9,10)
X = matrix(0,100,7)
y = rnorm(100,0,0.2)*sum(b)*rnorm(1,1,0.1)
for (i in 1:100)
{
  r<-rnorm(7,0,0.2)
  X[i,]<-r*b+b
}
lin.reg(y, X)
ridge.reg(y, X, 0.1)
lasso.reg(y, X, 0.1)
cross.validation(y,X,0.1,ridge.reg)
cross.validation(y,X,0.1,lasso.reg)

