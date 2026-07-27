source("wlrma.R")
source("awlrma.R")
library(dplyr)
library(tictoc)
library(tidyr)
library(Matrix)
library(fields)
library(matrixStats)

#####################################
#set parameters
#####################################

eps = 1e-6
maxiter = 1000

data = read.table('Data/ratings.dat', sep = ":") %>%
  dplyr::select(1,3,5) %>%
  rename(userId = 1, movieId = 2, rating = 3)
M = sparseMatrix(i = data$userId, j = data$movieId, x = data$rating, dims = c(max(data$userId), max(data$movieId)))
M = M[,colSums(M != 0) > 300]
M = M[rowSums(M != 0) > 200,]
min(colSums(M != 0))
min(rowSums(M != 0))
cat('data dim:', dim(M))
ind = which(M != 0, arr.ind = TRUE)

init = function(SVD, k){
  d = pmax(SVD$d, 0)
  A = SVD$u[,1:k] %*% diag(sqrt(d[1:k])) 
  B = SVD$v[,1:k] %*% diag(sqrt(d[1:k]))
  return(list(A = A, B = B))
}

nfold = 10
set.seed(1)
folds = sample(1:nfold, nrow(ind), replace = TRUE)
ks = seq(2, 20, 2)

AB = list()
result = c()

zero2nas = function(M){
  M = as.matrix(M) 
  M[M == 0] = NA
  return(M)
}

for(fold in 1:nfold){
  cat("\nfold: ", fold)
  train = (folds != fold)
  test = (folds == fold)
  
  Mtrain = sparseMatrix(i = ind[train,1], j = ind[train,2], x = M[ind[train,]], dims = dim(M))
  SVD = svd(as.matrix(Mtrain))
  
  #unit weights
  Wunit = sparseMatrix(i = ind[train,1], j = ind[train,2], x = 1, dims = dim(M))

  #fixed weights 1/s2
  s2 = rowSds(zero2nas(Mtrain), na.rm = T)^2
  w = min(s2)/s2
  Wfixed = sparseMatrix(i = ind[train,1], j = ind[train,2], x = w[ind[train,1]], dims = dim(M))

  for(k in ks){
    cat("\nk: ", k)
    init0 = init(SVD, k)

    cat("\nmethod: ", 1)
    wlrma = WLRMA(Mtrain, Wunit, type = "hard", k, method = "als", initialization = init0,
                 acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = F)
    E2 = sparseMatrix(i = ind[,1], j = ind[,2], x = (M[ind] - rowSums(wlrma$solution$A[ind[,1],] * wlrma$solution$B[ind[,2],]))^2, dims = dim(M))
    result = rbind(result, data.frame(loss = 1/2 * mean(E2[ind[test,]]), fold, k, weights = "unit"))
    AB[[paste0("k = ", k)]] = wlrma$solution

    cat(" ", 2)
    wlrma = WLRMA(Mtrain, Wfixed, type = "hard", k, method = "als", initialization = init0,
                  acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = F)
    E2 = sparseMatrix(i = ind[,1], j = ind[,2], x = (M[ind] - rowSums(wlrma$solution$A[ind[,1],] * wlrma$solution$B[ind[,2],]))^2, dims = dim(M))
    result = rbind(result, data.frame(loss = 1/2 * mean(E2[ind[test,]]), fold, k, weights = "pre-computed"))
    AB[[paste0("k = ", k)]] = wlrma$solution

    cat(" ", 3)
    tic()
    wlrma = AWLRMA(Mtrain, type = "hard", k, method = "als", initialization = init0,
                    acc_method = "nesterov", outerloop = list(threshold = eps, max_iter = maxiter, verbose = T),
                    innerloop = list(threshold = eps, max_iter = 100, verbose = F))
    toc()
    E2 = sparseMatrix(i = ind[,1], j = ind[,2], x = (M[ind] - rowSums(wlrma$solution$A[ind[,1],] * wlrma$solution$B[ind[,2],]))^2, dims = dim(M))
    result = rbind(result, data.frame(loss = 1/2 * mean(E2[ind[test,]]), fold, k, weights = "adaptive"))
    AB[[paste0("k = ", k)]] = wlrma$solution
    
    saveRDS(AB, paste0("Fits/real_data_hard_weights_AB.rds"))
    saveRDS(result, paste0("Fits/real_data_hard_weights_info.rds"))
  }
}

wlrma = WLRMA(Mtrain, Wfixed, type = "hard", k, method = "als", initialization = init0,
              acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = F)

#weights
s2 = rowSds(zero2nas(M), na.rm = T)^2

k = 10
SVD = svd(as.matrix(M))
init0 = init(SVD, k)
wlrma = AWLRMA(M, type = "hard", k, method = "als", initialization = init0,
               acc_method = "nesterov", outerloop = list(threshold = eps, max_iter = maxiter, verbose = T),
               innerloop = list(threshold = eps, max_iter = 100, verbose = T))

rbind(data.frame(s2 = s2, weights = "pre-computed"),
      data.frame(s2 = wlrma$s2, weights = "adaptive")) %>%
  saveRDS(paste0("Fits/real_data_hard_weights_s2.rds"))
  