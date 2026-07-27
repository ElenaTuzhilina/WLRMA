source("wlrma.R")

loglik = function(X, Z, W, parameter, type){
  loss = sum(W * (log(1 + exp(X)) - Z * X)) 
  if(type == "soft")  loss = loss + parameter * npmr::nuclear(X)
  return(loss/lenght(Z))
}

loglik_als = function(A, B, Z, W, parameter, type){
  X = A %*% t(B)
  loss = sum(W * (log(1 + exp(X)) - Z * X)) 
  if(type == "soft") loss = loss + parameter/2 * (sum(A^2)+ sum(B^2))
  return(loss/length(Z))
}

SOA = function(X, Z, W){
  E = exp(X)
  H = E/(1 + E)^2 
  M = X - 1 - E + Z/pmax(H, 1e-6)
  #cat("E", range(E), "H", range(H), "M", range(M))
  return(list(M = M, W = H * W))
}

mat2probs = function(X){
  E = exp(X)
  P = E/(1 + E)
  return(P)
}

LWLRMA = function(Z, W, type = "hard", parameter, method = "svd", initialization = NULL, 
                  acc_method = "baseline", acc_parameters = list(depth = 3, delay = 0, guarded = FALSE, reg_depth = 3, gamma = 0), 
                  outerloop = list(threshold = 1e-8, max_iter = 100, verbose = TRUE),
                  innerloop = list(threshold = 1e-8, max_iter = 100, verbose = FALSE)){
  n = nrow(Z)
  p = ncol(Z)
  
  solution = initialization
  epoch = 0
  delta = Inf
  obj = Inf
  info  = c()
  iter = Inf
  info_names = c('epoch', 'iter', 'loss', 'delta')
  
  while(delta > outerloop$threshold & epoch < outerloop$max_iter & iter > 1){
    epoch = epoch + 1
    
    #save
    obj0 = obj
    
    if(method == "svd") X = solution$X
    if(method == "als") X = solution$A %*% t(solution$B)
    soa = SOA(X, Z, W)
    W0 = W * soa$W
    wlrma = WLRMA(soa$M, W0/max(W0), type, parameter, method, initialization = solution, acc_method, acc_parameters, innerloop$threshold, innerloop$max_iter, innerloop$verbose)
    solution = wlrma$solution
    if(method == "svd") obj = loglik(solution$X, Z, W, parameter, type)
    if(method == "als") obj = loglik_als(solution$A, solution$B, Z, W, parameter, type)
    
    iter = max(wlrma$info$iter)
    if(epoch > 1) delta = abs((obj - obj0)/obj0)
    res = data.frame(epoch, iter, obj, delta)
    colnames(res) = info_names
    info = rbind(info, res)
    if(outerloop$verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  solution = wlrma$solution
  return(list(solution = solution, info = info))
}
