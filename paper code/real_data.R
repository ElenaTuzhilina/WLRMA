#library(WLRMA)
source("wlrma.R")
library(dplyr)
library(tictoc)
library(tidyr)
library(Matrix)

#####################################
#set parameters
#####################################

eps = 1e-8
m = 3
d = 3
maxiter = 1000
guarded = T

data = read.table('Data/ratings.dat', sep = ":") %>%
  dplyr::select(1,3,5) %>%
  rename(userId = 1, movieId = 2, rating = 3)
M = sparseMatrix(i = data$userId, j = data$movieId, x = data$rating, dims = c(max(data$userId), max(data$movieId)))
M = M[, colSums(M != 0) > 0]
ind = which(M != 0, arr.ind = TRUE)
W = sparseMatrix(i = ind[,1], j = ind[,2], x = 1, dims = dim(M))
cat('data dim:', dim(M))

#####################################
#functions
#####################################

compare_convergence = function(M, W, parameters, type, init){
  result = c()
  AB1 = list()
  AB2 = list()
  AB3 = list()
  for(parameter in parameters){
    init0 = init(parameter)
    sol1 = WLRMA(M, W, type = type, parameter = parameter, method = "als", initialization = list(A = init0$A, B = init0$B), acc_method = "baseline", threshold = eps, max_iter = maxiter, verbose = TRUE)
    sol2 = WLRMA(M, W, type = type, parameter = parameter, method = "als", initialization = list(A = init0$A, B = init0$B), acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = TRUE)
    sol3 = WLRMA(M, W, type = type, parameter = parameter, method = "als", initialization = list(A = init0$A, B = init0$B), acc_method = "anderson", acc_parameter = list(depth = m, delay = 0, guarded = guarded), threshold = eps, max_iter = maxiter, verbose = TRUE)
    result = rbind(result, data.frame(rbind(data.frame(sol1$info, method = "baseline"), 
                                            data.frame(sol2$info, method = "nesterov"), 
                                            data.frame(sol3$info, method = "anderson")), 
                                      type = type, parameter = parameter))
    AB1[[paste0("parameter = ", parameter)]] = sol1$solution
    AB2[[paste0("parameter = ", parameter)]] = sol2$solution
    AB3[[paste0("parameter = ", parameter)]] = sol3$solution
    #saveRDS(list("baseline" = AB1, "nesterov" = AB2, "anderson" = AB3), paste0("Fits/real_data_", type, "_AB.rds"))
    #saveRDS(result, paste0("Fits/real_data_", type, "_info.rds"))
  }
  return(list(info = result, AB = list("baseline" = AB1, "nesterov" = AB2, "anderson" = AB3))) 
}

############################################
#compare soft impute
############################################

init = function(lambda){
  SVD = readRDS("Fits/SVDofX.rds")
  if(lambda == 10) k = 600
  if(lambda == 20) k = 350
  if(lambda == 30) k = 250
  if(lambda == 40) k = 100
  if(lambda == 50) k = 50
  if(lambda == 100) k = 20
  d = pmax(SVD$d - lambda, 0)
  A = SVD$u[,1:k] %*% diag(sqrt(d[1:k])) 
  B = SVD$v[,1:k] %*% diag(sqrt(d[1:k]))
  return(list(A = A, B = B))
}

lambdas = c(100, 50, 40, 30, 20, 10)
result = compare_convergence(M, W, lambdas, "soft", init)

############################################
# find df
############################################

AB = readRDS("Fits/real_data_soft_AB_full.rds")

compute_df = function(W, lambda, A, B){
  p = ncol(W)
  dfs = rep(0, p)
  for(i in 1:p){
    w = W[,i]
    H = t(A[W[,i] == 1,]) %*% A[W[,i] == 1,]
    Hlambda = H
    diag(Hlambda) = diag(Hlambda) + lambda
    dfs[i] = sum(diag(solve(Hlambda) %*% H))
  }
  dfs
}

dfsA = matrix(0, length(lambdas), ncol(M))
dfsB = matrix(0, length(lambdas), nrow(M))
for(i in 1:length(lambdas)){
  lambda = lambdas[i]
  cat(lambda)
  dfsA[i,] = compute_df(W, lambda, AB[["anderson"]][[paste0("parameter = ", lambda)]]$A, AB[["anderson"]][[paste0("parameter = ", lambda)]]$B)
  dfsB[i,] = compute_df(t(W), lambda, AB[["anderson"]][[paste0("parameter = ", lambda)]]$B, AB[["anderson"]][[paste0("parameter = ", lambda)]]$A)
}

rbind(data.frame(dfs = c(dfsA), lambda = lambdas, type = "A"), data.frame(dfs = c(dfsB), lambda = lambdas, type = "B")) %>%
  saveRDS("Fits/real_data_dfs.rds")


############################################
# find df and compare to hard
############################################

init = function(k){
  SVD = readRDS("Fits/SVDofX.rds")
  d = SVD$d
  A = SVD$u[,1:k, drop = F] %*% diag(sqrt(d[1:k]), k, k) 
  B = SVD$v[,1:k,  drop = F] %*% diag(sqrt(d[1:k]), k, k)
  return(list(A = A, B = B))
}

#lambdas = c(20, 30, 40, 50)
ks = c(3, 5, 12, 29)
result = compare_convergence(M, W, ks, "hard", init)

############################################
#hard impute with different gamma
############################################

init = function(k){
  SVD = readRDS("Fits/SVDofX.rds")
  d = SVD$d
  A = SVD$u[,1:k, drop = F] %*% diag(sqrt(d[1:k]), k, k) 
  B = SVD$v[,1:k,  drop = F] %*% diag(sqrt(d[1:k]), k, k)
  return(list(A = A, B = B))
}

compare_convergence_reg = function(M, W, parameters, type, init){
  result = c()
  AB = list()
  for(parameter in parameters){
    init0 = init(parameter)
    for(gamma in gammas){
      sol = WLRMA(M, W, type = type, parameter = parameter, method = "als", initialization = list(A = init0$A, B = init0$B), acc_method = "randerson", acc_parameter = list(depth = 3, delay = 0, guarded = guarded, reg_depth = d, gamma = gamma), threshold = eps, max_iter = maxiter, verbose = TRUE)
      result = rbind(result, data.frame(sol$info, method = "anderson", type = type, parameter = parameter, gamma = gamma))
      saveRDS(result, paste0("Fits/real_data_", type, "_reg_info.rds"))
    }
  }
  return(result)
}

ks = c(3, 5, 12, 29)
gammas = c(0.001, 0.01, 0.1, 1)
result = compare_convergence_reg(M, W, ks, "hard", init)

