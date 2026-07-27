source("wlrma.R")
source("lwlrma.R")
library(dplyr)
library(tictoc)
library(tidyr)
library(Matrix)
library(fields)
library(matrixStats)
library(ggplot2)
library(pROC)

#####################################
#set parameters
#####################################

eps = 1e-5
maxiter = 1000
type = "hard"

data = read.table('Data/ratings.dat', sep = ":") %>%
  dplyr::select(1,3,5) %>%
  rename(userId = 1, movieId = 2, rating = 3)
M = sparseMatrix(i = data$userId, j = data$movieId, x = data$rating, dims = c(max(data$userId), max(data$movieId)))
M = M[,colSums(M != 0) > 300]
M = M[rowSums(M != 0) > 200,]
cat('data dim:', dim(M))
n = nrow(M)
p = ncol(M)
M = (as.matrix(M != 0) * 1)

init = function(SVD, k){
  d = pmax(SVD$d, 0)
  A = SVD$u[,1:k] %*% diag(sqrt(d[1:k])) 
  B = SVD$v[,1:k] %*% diag(sqrt(d[1:k]))
  return(list(A = A, B = B))
}

nfold = 10
set.seed(1)
folds = sample(1:nfold, length(M), replace = TRUE)
ks = seq(5, 50, 5)

AB = list()
rocs = list()
result = c()

for(fold in 1:nfold){
  cat("\nfold: ", fold)
  train = (folds != fold)
  test = (folds == fold)
  
  SVD = svd(as.matrix(Mtrain))
  Wtrain = matrix(0, n, p)
  Wtrain[train] = 1
  Wtest = matrix(0, n, p)
  Wtest[test] = 1
  
  for(k in ks){
    cat("\nk: ", k)
    init0 = init(SVD, k)
    
    wlrma = LWLRMA(M, Wtrain, type, k, method = "als", initialization = init0,
                    acc_method = "nesterov", outerloop = list(threshold = eps, max_iter = maxiter, verbose = T),
                    innerloop = list(threshold = eps, max_iter = 100, verbose = F))
    A = wlrma$solution$A
    B = wlrma$solution$B
    X = A %*% t(B)
    
    suppressMessages({ 
      AUC = auc(M[test], mat2probs(X)[test]) 
      ROC = roc(M[test], mat2probs(X)[test])})
    result = rbind(result, data.frame(loss = loglik_als(A, B, M, Wtest, parameter, type), auc = AUC, fold, k))
    rocs[[paste0("k = ", k)]] = ROC
    AB[[paste0("k = ", k)]] = wlrma$solution
    
    saveRDS(rocs, paste0("Fits/real_data_hard_prob_ROC.rds"))
    saveRDS(AB, paste0("Fits/real_data_hard_prob_AB.rds"))
    saveRDS(result, paste0("Fits/real_data_prob_info.rds"))
  }
}

SVD = svd(as.matrix(M))
init0 = init(SVD, k)
wlrma = LWLRMA(M, W = 1, type, 15, method = "als", initialization = init0,
               acc_method = "nesterov", outerloop = list(threshold = eps, max_iter = maxiter, verbose = T),
               innerloop = list(threshold = eps, max_iter = 100, verbose = F))
A = wlrma$solution$A
B = wlrma$solution$B
X = A %*% t(B)
P = mat2probs(X)
data.frame(expand.grid(movie = 1:p, user = 1:n), probability = c(P), observed = c(M)) %>%
  saveRDS(paste0("Fits/real_data_hard_logit_probs.rds"))
