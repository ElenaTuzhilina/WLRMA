source("wlrma.R")

AWLRMA = function(M, type = "hard", parameter, method = "svd", initialization = NULL, 
                  acc_method = "baseline", acc_parameters = list(depth = 3, delay = 0, guarded = FALSE, reg_depth = 3, gamma = 0), 
                  outerloop = list(threshold = 1e-8, max_iter = 100, verbose = TRUE),
                  innerloop = list(threshold = 1e-8, max_iter = 100, verbose = FALSE)){
  n = nrow(M)
  p = ncol(M)
  if(class(M)[1] == "dgCMatrix") sparse = TRUE
  else sparse = FALSE
  
  if(sparse){
    ind = which(M != 0, arr.ind = TRUE)
    obs = rowSums(M != 0)
  } else {
    ind = which(!is.na(M), arr.ind = TRUE)
    obs = rowSums(!is.na(M))
    W = matrix(0, dim(M))
  }
  
  w = 1/rowSds(zero2nas(M), na.rm = T)^2
  w = w/max(w)
  epoch = 0
  delta = Inf
  obj = Inf
  info  = c()
  iter = Inf
  #bad = c()
  info_names = c('epoch', 'iter', 'loss', 'delta')
  
  while(delta > outerloop$threshold & epoch < outerloop$max_iter & iter > 1){
    epoch = epoch + 1
    
    #save
    obj0 = obj
    
    if(sparse) W = sparseMatrix(i = ind[,1], j = ind[,2], x = w[ind[,1]], dims = dim(M))
    else W[ind] = w[ind[,1]]
    
    wlrma = WLRMA(M, W, type, parameter, method, initialization, acc_method, acc_parameters, innerloop$threshold, innerloop$max_iter, innerloop$verbose)
    
    if(method == "svd"){
      X = wlrma$solution$X
      s2 = rowSums((M - X)^2)/obs
      obj = loss(X, M, W, parameter, type)
    } 
    if(method == "als"){
      A = wlrma$solution$A
      B = wlrma$solution$B
      if(sparse){ 
        E2 = sparseMatrix(i = ind[,1], j = ind[,2], x = (M[ind] - rowSums(A[ind[,1],] * B[ind[,2],]))^2, dims = dim(M))
        obj = 1/2 * sum(W * E2)/ length(M)
      } else {
        E2 = (M - A %*% t(B))^2
        obj = loss_als(A, B, M, W, parameter, type)
      }
      s2 = rowSums(E2, na.rm = T)/obs
    } 
    s20 = s2
    hist(s2)
    #newbad =  which(s2 < 1e-3)
    #bad = union(bad, newbad)
    #if(length(newbad) > 0) cat(" bad samples", bad)
    #s2[newbad] = min(s2[-newbad])
    #hist(s2)
    s2 = pmax(s2, 1e-3)
    w = 1/s2
    w = w/max(w)
    initialization = wlrma$solution
    
    iter = max(wlrma$info$iter)
    if(epoch > 1) delta = abs((obj - obj0)/obj0)
    res = data.frame(epoch, iter, obj, delta)
    colnames(res) = info_names
    info = rbind(info, res)
    if(outerloop$verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  solution = wlrma$solution
  return(list(solution = solution, W = W, s2 = s20, info = info))
}
