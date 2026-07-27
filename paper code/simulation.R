#library(WLRMA)
library(dplyr)
library(ggplot2)
library(latex2exp)
library(tidyr)
library(tictoc)
library(Matrix)
source("wlrma.R")

#####################################
#set parameters
#####################################

n = 1000
p = 100
eps = 1e-8
maxiter = 300
guarded = F

#####################################
#functions
#####################################

compare_convergence = function(X, W, parameters, type, init){
  result = c()
  for(parameter in parameters){
    X0 = init(parameter)
    sol1 = WLRMA(X, W, type = type, parameter = parameter, method = "svd", initialization = list(X = X0), acc_method = "baseline", threshold = eps, max_iter = maxiter, verbose = TRUE)
    sol2 = WLRMA(X, W, type = type, parameter = parameter, method = "svd", initialization = list(X = X0), acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = TRUE)
    sol3 = WLRMA(X, W, type = type, parameter = parameter, method = "svd", initialization = list(X = X0), acc_method = "anderson", acc_parameter = list(depth = m, delay = 0, guarded = guarded), threshold = eps, max_iter = maxiter, verbose = TRUE)
    result = rbind(result, data.frame(rbind(data.frame(sol1$info, method = "baseline"), 
                                            data.frame(sol2$info, method = "nesterov"), 
                                            data.frame(sol3$info, method = "anderson")), 
                                      type = type, parameter = parameter))
  }
  return(result)
}

generateX = function(r, sigma){
  set.seed(123)
  A = matrix(rnorm(n*r, 0, 1), n, r)
  B = matrix(rnorm(p*r, 0, 1), p, r)
  E = matrix(rnorm(n*p, 0, sigma), n, p)
  X = A %*% t(B) + E
  return(X)
}

generateW = function(){
  set.seed(123)
  W = matrix(runif(n*p), n, p)
  W = (W - min(W))/(max(W) - min(W))
  return(W)
}

#####################################
#compare soft and hard impute
#####################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
m = 3
ks = c(10, 25, 50, 75)
lambdas = c(150, 100, 30, 5)

#compare hard impute
init = function(k) matrix(0, nrow(X), ncol(X))
resulthard = compare_convergence(X, W, ks, "hard", init)

# #select lambdas
# resulthard %>% group_by(parameter) %>% summarise(min(loss_no_penalty))
# sol2 = WLRMA(X, W, type = "soft", parameter = lambda, method = "svd", initialization = list(X = X0), acc_method = "nesterov", threshold = eps, max_iter = maxiter, verbose = TRUE)

#compare soft impute
init = function(k) matrix(0, nrow(X), ncol(X))
resultsoft = compare_convergence(X, W, lambdas, "soft", init)

saveRDS(resulthard, "Fits/simulation_hard.rds")
saveRDS(resultsoft, "Fits/simulation_soft.rds")

#####################################
#hard impute with different initializations
#####################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
m = 3
ks = c(10, 25, 50, 75)

#zero
init = function(k) matrix(0, nrow(X), ncol(X))
resultinit = data.frame(compare_convergence(X, W, ks, "hard", init), init = "zero")

#warm start
init = function(k){
  SVD = svd(X)
  SVD$u[,1:k] %*% diag(SVD$d[1:k]) %*% t(SVD$v[,1:k])
}
resultinit = rbind(resultinit, data.frame(compare_convergence(X, W, ks, "hard", init), init = "warm"))

#random low-rank
init = function(k){
  A0 = matrix(rnorm(n*k, 0, 1), n, k)
  B0 = matrix(rnorm(p*k, 0, 1), p, k)
  A0 %*% t(B0)
}
resultinit = rbind(resultinit, data.frame(compare_convergence(X, W, ks, "hard", init), init = "low-rank"))

#random 
init = function(k) matrix(rnorm(n*p, 0, 1), n, p)
resultinit = rbind(resultinit, data.frame(compare_convergence(X, W, ks, "hard", init), init = "random"))

saveRDS(resultinit, "Fits/simulation_hard_init.rds")

############################################
#soft impute with different SNRs and ranks
############################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
m = 3
lambdas = c(150, 100, 30, 5)
rs = c(10, 25, 50, 75)
sigmas = c(1, 5, 10, 20)

init = function(lambda){
  SVD = svd(X)
  SVD$u %*% diag(pmax(SVD$d - lambda, 0)) %*% t(SVD$v)
}
resultrk = c()
for(r in rs){
  for(sigma in sigmas){
    A = matrix(rnorm(n*r, 0, 1), n, r)
    B = matrix(rnorm(p*r, 0, 1), p, r)
    E = matrix(rnorm(n*p, 0, sigma), n, p)
    X = A %*% t(B) + E
    resultrk = rbind(resultrk, data.frame(compare_convergence(X, W, lambdas, "soft", init), r, sigma))
  }
}

saveRDS(resultrk, "Fits/simulation_soft_sigma_rank.rds")

############################################
#soft impute with different depth for anderson
############################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
lambdas = c(150, 100, 30, 5)
ms = c(2, 3, 5, 10)

init = function(lambda){
  SVD = svd(X)
  SVD$u %*% diag(pmax(SVD$d - lambda, 0)) %*% t(SVD$v)
}
resultaa = c()
for(lambda in lambdas){
  for(m in ms){
    X0 = init(lambda)
    sol = WLRMA(X, W, type = "soft", parameter = lambda, method = "svd", initialization = list(X = X0), acc_method = "anderson", acc_parameter = list(depth = m, delay = 0, guarded = guarded), threshold = eps, max_iter = maxiter, verbose = TRUE)
    resultaa = rbind(resultaa, data.frame(sol$info, parameter = lambda, depth = m))
  }
}

saveRDS(resultaa, "Fits/simulation_soft_depth.rds")

############################################
#hard impute with different gamma
############################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
#ks = c(10, 25, 50, 75)
ks = 50
ds = c(3, 5, 10)
m = 3
gammas = c(0.0001, 0.001, 0.01, 0.1, 1, 10)

resultraa = c()
for(k in ks){
  init = function(k) matrix(0, nrow(X), ncol(X))
  X0 = init(k)
  sol = WLRMA(X, W, type = "hard", parameter = k, method = "svd", initialization = list(X = X0), acc_method = "anderson", acc_parameter = list(depth = m, delay = 0, guarded = guarded), threshold = eps, max_iter = maxiter, verbose = TRUE)
  resultraa = rbind(resultraa, merge(data.frame(sol$info, k = k, gamma = 0), data.frame(reg_depth = ds), all = T))
  coefraa = data.frame(sol$coefs, iter = 1:nrow(sol$coefs), gamma = 0) %>% pivot_longer(!c(gamma, iter), names_to = "coef", values_to = "value") 
  for(d in ds){
    for(gamma in gammas){
        X0 = init(k)
        sol = WLRMA(X, W, type = "hard", parameter = k, method = "svd", initialization = list(X = X0), acc_method = "randerson", acc_parameter = list(depth = m, delay = 0, guarded = guarded, reg_depth = d, gamma = gamma), threshold = eps, max_iter = maxiter, verbose = TRUE)
        resultraa = rbind(resultraa, data.frame(sol$info, gamma = gamma, k = k, reg_depth = d))
        if(d == 3) coefraa = rbind(coefraa, data.frame(sol$coefs, iter = 1:nrow(sol$coefs), gamma = gamma) %>% pivot_longer(!c(gamma, iter), names_to = "coef", values_to = "value")) 
    }
  }
}

saveRDS(resultraa, "Fits/simulation_hard_reg.rds")
saveRDS(coefraa, "Fits/simulation_hard_reg_coef.rds")


############################################
#degrees-of-freedom
############################################

#generate data
X = generateX(r = 75, sigma = 1)
W = generateW()

#set parameters
lambdas = c(200, 175, 150, 125, 100, 75, 50, 25)
rs = c(50, 50, 50, 60, 60, 80, 80, 100)

compute_df = function(W, lambda, A, B){
  p = ncol(W)
  dfs = rep(0, p)
  for(i in 1:p){
    H = t(A) %*% diag(W[,i]) %*% A
    Hlambda = H
    diag(Hlambda) = diag(Hlambda) + lambda
    dfs[i] = sum(diag(solve(Hlambda) %*% H))
  }
  dfs
}

loss_no_penalty = function(A, B){
  mean(W * (X - A %*% t(B)))^2
}


dfsA = matrix(0, length(lambdas), p)
dfsB = matrix(0, length(lambdas), n)
dfs = rep(0, length(lambdas))
resultdf = c()

for(i in 1:length(lambdas)){
  r = rs[i]
  lambda = lambdas[i]
  sols = WLRMA(X, W, type = "soft", parameter = lambda, method = "als", initialization = list(A = matrix(rnorm(n*r, 0, 1), n, r), B = matrix(rnorm(p*r, 0, 1), p, r)), acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  
  rank = min(sols$info$rank, na.rm = T)
  solh = WLRMA(X, W, type = "hard", parameter = rank, method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  
  dfsA[i,] = compute_df(W, lambda, sols$solution$A, sols$solution$B)
  solhAmean = WLRMA(X, W, type = "hard", parameter = round(mean(dfsA[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  solhAmin = WLRMA(X, W, type = "hard", parameter = round(min(dfsA[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  solhAmax = WLRMA(X, W, type = "hard", parameter = round(max(dfsA[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  
  dfsB[i,] = compute_df(t(W), lambda, sols$solution$B, sols$solution$A)
  solhBmean = WLRMA(X, W, type = "hard", parameter = round(mean(dfsB[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  solhBmin = WLRMA(X, W, type = "hard", parameter = round(min(dfsB[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  solhBmax = WLRMA(X, W, type = "hard", parameter = round(max(dfsB[i,])), method = "als", initialization = NULL, acc_method = "baseline", threshold = eps, max_iter = 1000, verbose = TRUE)
  
  resultdf = rbind(resultdf, data.frame(lambda = lambda,
  rank  = c(rank, rank, mean(dfsA[i,]), mean(dfsB[i,])),
  rank_min = c(rank, rank, min(dfsA[i,]), min(dfsB[i,])),
  rank_max = c(rank, rank, max(dfsA[i,]), max(dfsB[i,])),
  loss = c(min(sols$info$loss_no_penalty), min(solh$info$loss_no_penalty), min(solhAmean$info$loss_no_penalty), min(solhBmean$info$loss_no_penalty)), 
  loss_min = c(min(sols$info$loss_no_penalty), min(solh$info$loss_no_penalty), min(solhAmin$info$loss_no_penalty), min(solhBmin$info$loss_no_penalty)),
  loss_max = c(min(sols$info$loss_no_penalty), min(solh$info$loss_no_penalty), min(solhAmax$info$loss_no_penalty), min(solhBmax$info$loss_no_penalty)),
  method = c("soft", "hard with rank", "hard with df (A fixed)", "hard with df (B fixed)")))
}

saveRDS(list(info = resultdf, 
             dfs = rbind(data.frame(dfsA, lambda = lambdas) %>% pivot_longer(!lambda) %>% select(-name) %>% mutate(dfby = "A"),
                                          data.frame(dfsB, lambda = lambdas) %>% pivot_longer(!lambda) %>% select(-name) %>% mutate(dfby = "B"))), 
        "Fits/simulation_df.rds")
