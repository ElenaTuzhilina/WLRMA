#library(WLRMA)
source("wlrma.R")
source("awlrma.R")
library(dplyr)
library(tictoc)
library(tidyr)
library(Matrix)
library(fields)

#####################################
#set parameters
#####################################

n = 1000
p = 100
eps = 1e-8
maxiter = 200
guarded = F
eps = 1e-5

generateM = function(r, sigma, prop, seed = 123){
  set.seed(seed)
  A = matrix(rnorm(n*r, 0, 1), n, r)
  B = matrix(rnorm(p*r, 0, 1), p, r)
  sigma = runif(n)
  E = matrix(rnorm(n*p, 0, 1), n, p)
  E = t(scale(t(E), center = T, scale = 1/sigma))
  M = A %*% t(B) + E
  nas = sample(1:(n*p), n*p*prop)
  Mtrain = M
  Mtrain[nas] = NA
  Mtest = M
  Mtest[-nas] = NA
  return(list(Mtrain = Mtrain, Mtest = Mtest))
}

type = "hard"
props = seq(0.05, 0.3, 0.05)
ks = c(5, 10, 20, 30)
seeds = 1:30
result = c()

init = function(X, k){
  X[is.na(X)] = 0
  SVD = svd(X)
  SVD$u[,1:k] %*% diag(SVD$d[1:k]) %*% t(SVD$v[,1:k])
}

for(seed in seeds){
  cat("\n\nseed:", seed)
  for(prop in props){
    gen = generateM(r = 20, sigma = 10, prop = prop, seed)
    cat("\n\nproportion:", prop)
    for(k in ks){
      cat("\nk: ", k)
      wlrma = WLRMA(gen$Mtrain, W = NULL, type, k, method = "svd", initialization = list(X = init(gen$Mtrain, k)), 
                    acc_method = "baseline", threshold = eps, maxiter, verbose = F)
      awlrma = AWLRMA(gen$Mtrain, type, k, method = "svd", initialization = list(X = init(gen$Mtrain, k)),  
                      acc_method = "baseline", outerloop = list(threshold = eps, max_iter = maxiter, verbose = T), 
                      innerloop = list(threshold = eps, max_iter = maxiter, verbose = F))
      
      result = rbind(result, data.frame(loss = c(1/2 * mean((wlrma$solution$X - gen$Mtest)^2, na.rm = T), 
                                                 1/2 * mean((awlrma$solution$X - gen$Mtest)^2, na.rm = T)), 
                 weights = c("fixed", "adaptive"), seed, prop, k))
    }
    saveRDS(result, "Fits/simulation_hard_adaptive.rds")
  }
}



