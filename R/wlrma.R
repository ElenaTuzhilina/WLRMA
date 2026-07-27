#' @title Weighted Low-Rank Matrix Approximation
#' 
#' @description WLRMA function solves a general weighted low-rank matrix approximation problem. Given some matrix \eqn{M} and a matrix of weights \eqn{W} 
#' (both of size \eqn{n x p}) as well as some rank \eqn{k}, WLRMA seeks for matrix \eqn{X} such that \eqn{rk(X)\le k} and 
#' that minimizes the weighted frobenious norm
#' \deqn{||\sqrt W * (M - X)||_F^2           (1)} 
#' Here \eqn{*} corresponds to the element-wise matrix product.
#' The corresponding convex relaxation of this problem is 
#' \deqn{1/2 ||\sqrt W * (M - X)||_F^2 + \lambda||X||_*            (2)}
#' Here \eqn{||...||_*} refers to the matrix nuclear norm.
#' 
#' These two problems can be solved via an algorithm based on projected gradeint descent (PGD), 
#' which requires taking singular value decomposition of \eqn{n x p} matrix at each iteration. 
#' Although when \eqn{n} and \eqn{p} are small this can be quite an efficient approach, in high dimensions calculating SVD could be impossible.
#' As an alternative for high-dimensional data one can consider the algorithm based on the alternating least squares (ALS). 
#' This algorithm utilizes the fact that any solution \eqn{X} of rank \eqn{k} can be decomposed in the 
#' product \eqn{X = AB^T} where \eqn{A} is \eqn{n x k} matrix
#' and \eqn{B} is \eqn{p x k} matrix. Thus the ALS approach finds \eqn{A} and \eqn{B} that minimize
#' \deqn{||\sqrt W * (M - AB^T)||_F^2            (3)}
#' To build the convex relaxation for this problem one can use the fact that for each \eqn{\lambda} there exists 
#' \eqn{r} such that for all \eqn{k\ge r}
#' problem (2) is equivalent to minimizing 
#' \deqn{1/2 ||\sqrt W  * (M - AB^T)||_F^2 + \lambda/2 ||A||_F^2 + \lambda/2 ||B||_F^2         (4)}
#' with respect to \eqn{A} and \eqn{B}.
#' 
#' @param M a rectangular \eqn{n x m} matrix the approximation should be computed for. 
#' @param W a rectangular \eqn{n x m} matrix containing non-negative weights. By default \code{W = NULL}, i.e. non-weighted problem is solved.
#' @param type the problem type. Default value is \code{type = "soft"}, i.e. the WLRMA convex relaxation (2) is solved. Use \code{type = "hard"} to solve the non-convex WLRMA problem (1). 
#' @param parameter the problem hyperparameter. For \code{type = "hard"}, i.e. non-convex WLRMA, \code{parameter} corresponds to the solution rank \eqn{k}. For \code{type = "soft"}, i.e. the convex relaxation, \code{parameter} corresponds to \eqn{\lambda}.
#' @param method the method the solution is acheved. By default \code{method = "svd"}, thus the algorithm based on projected gradient descent
#' and utilizing SVD at each iteration is used. If \code{method = "als"} the alternating least squares algorithm is used to solve the ploblem.
#' @param initialization a list containing initialization for the algorithm. If PGD-type algorithm is used the list should contain the inialization for \eqn{X}, i.e. 
#' \code{initialization = list(X = ...)}.
#' If ALS-type algorithm is used the list should contain the initialization for \eqn{A} and \eqn{B}, i.e. 
#' \code{initialization = list(A = ..., B = ...)}. 
#' The default value is \code{NULL} and random initialization is applied.
#' @param acc_method the acceleration method applied to improve the convergence speed. By default \code{acc_method = "baseline"} which corresponds to no acceleration. 
#' One can set \code{acc_method = "nesterov"} to use Nesterov acceleration. Alternatively, set \code{acc_method = "anderson"} to apply Anderson acceleration.
#' Option \code{acc_method = "randerson"} performs regularized Anderson acceleration.
#' @param acc_parameters a list containing acceleration parameters. For anderson acceleration set \code{acc_parameter = list(depth = ..., delay = ..., guarded = ...)}.
#' Here \code{depth} is the anderson depth, \code{delay} is the number of iterations the acceleration should be delayed by and \code{guarded} is logical variable indicating 
#' if guarded acceleration should be used.  
#' The default values are \code{depth = 3}, \code{delay = 0} and \code{guarded = FALSE}.
#' For regularized Anderson acceleration use two extra parameters \code{reg_depth = ...} and \code{gamma = ...} to control the regularization depth, i.e.
#' and regularization strength. 
#' @param threshold the convergence threshold. The algorithm stops when the relative change in the loss becomes less than \code{threshold}.
#' By default \code{threshold = 1e-8}.  
#' @param max_iter the maximum number of iterations performed. By default \code{max_iter = 100}.  
#' @param verbose logical variable indicating if progress after each iteration should be printed. By default \code{verbose = FALSE}.
#' @return A list containing:
#' \itemize{
#'   \item \code{solution} -- the list containing the solution of the optimization problem. For the PGD-type algorithm \code{solution = list(X = ...)}, 
#'   for the ALS-type \code{solution = list(A = ..., B = ...)}.
#'   \item \code{info} -- the data frame containing the algorithm progress while converging. The data frame includes: iteration number, 
#'   time, value of the loss, rank, relative change in the loss and relative change in the solution.
#' }
#' If Anderson acceleration was applied the function also returns
#' \itemize{
#'   \item \code{coefs} -- a matrix containing (column-wise) the paths for Anderson coefficients \eqn{\alpha}. 
#' }

#' @examples
#' #generate some data
#' set.seed(1)
#' n = 1000
#' p = 100
#' k = 70
#' A_true = matrix(rnorm(n * k), n, k)
#' B_true = matrix(rnorm(p * k), p, k)
#' noise = matrix(rnorm(n * p, 0, 1), n, p)
#' M = A_true %*% t(B_true) + noise
#' W = matrix(runif(n * p), n, p)
#' 
#' #solve non-convex WLRMA
#' type = "hard"
#' #find solution via PGD
#' method = "svd"
#' #set solution rank to 50
#' k = 50
#' sol = WLRMA(M, W, type, parameter = k, method, verbose = TRUE)
#' #plot the loss relative change vs. time
#' library(ggplot2)
#' ggplot(sol$info, aes(time, log(delta, 10))) +
#' geom_line() + ylab(bquote(log(Delta)))+
#' xlab("iteration")
#' 
#' #solve convex relaxation via PGD
#' type = "soft"
#' #set penalty factor to 30
#' lambda = 30
#' sol = WLRMA(M, W, type, parameter = lambda, method, verbose = TRUE)
#' #plot the rank vs. iteration
#' ggplot(sol$info, aes(iter, rank)) +
#' geom_line() + xlab("iteration")
#' 
#' #accelerate via Nesterov
#' nsol = WLRMA(M, W, type, parameter = lambda, method, acc_method = "nesterov", verbose = TRUE)
#' #compare convergence speed 
#'df = rbind(data.frame(sol$info, "acc_method" = "baseline"),
#'           data.frame(nsol$info, "acc_method" = "nesterov"))
#'ggplot(df, aes(time, log(delta, 10), color = acc_method)) +
#'  geom_line() + ylab(bquote(log(Delta)))+
#'  xlab("iteration")
#'  
#' #accelerate Anderson and regularized Anderson
#'  asol = WLRMA(M, W, type, parameter = lambda, method, acc_method = "anderson", verbose = TRUE)
#' rasol = WLRMA(M, W, type, parameter = lambda, method, acc_method = "randerson", verbose = TRUE)
#' #compare all four methods
#' df = rbind(df,
#'           data.frame(asol$info, "acc_method" = "anderson"),
#'           data.frame(rasol$info, "acc_method" = "randerson"))
#' ggplot(df, aes(time, log(delta, 10), color = acc_method)) +
#'  geom_line() + ylab(bquote(log(Delta)))+
#'  xlab("iteration")
#' 
#' #now generate large data
#' set.seed(1)
#' n = 1000
#' p = 1000
#' k = 30
#' A_true = matrix(rnorm(n * k), n, k)
#' B_true = matrix(rnorm(p * k), p, k)
#' noise = matrix(rnorm(n * p, 0, 1), n, p)
#' M = A_true %*% t(B_true) + noise
#' W = matrix(runif(n * p), n, p)
#' 
#' #solve the WLRMA convex relaxation via PGD method
#' type = "soft"
#' method = "svd"
#' lambda = 100
#' #set initialization
#' init = list(X = matrix(rnorm(n * p), n, p))
#' sol = WLRMA(M, W, type, parameter = lambda, method, initialization = init, verbose = TRUE)
#' 
#' #use ALS to solve the problem
#' method = "als"
#' #set initialization
#' r = 50
#' init = list(A = matrix(rnorm(n * r), n, r), B = matrix(rnorm(p * r), p, r))
#' sol.als = WLRMA(M, W, type, parameter = lambda, method, initialization = init, verbose = TRUE)
#' #compare the convergence speed
#' df = rbind(data.frame(sol$info, "method" = "svd"),
#' data.frame(sol.als$info, "method" = "als"))
#' ggplot(df, aes(time, log(delta, 10), color = method)) +
#'  geom_line() + ylab(bquote(log(Delta)))+
#'  xlab("iteration")
#' 
#' 
#' @export WLRMA
#' 

WLRMA = function(M, W = NULL, type = "soft", parameter, method = "svd", initialization = NULL, 
                 acc_method = "baseline", acc_parameters = list(depth = 3, delay = 0, guarded = FALSE, reg_depth = 3, gamma = 0), 
                 threshold = 1e-8, max_iter = 100, verbose = FALSE){
  n = nrow(M)
  p = ncol(M)
  if(is.null(W)) W = matrix(1, n, p)
  if(method == "svd"){
    if(is.null(initialization)) initialization = list(X = matrix(rnorm(n * p), n, p))
    if(acc_method == "baseline") solution = baseline(M, W, parameter, type, initialization, threshold, max_iter, verbose)
    if(acc_method == "nesterov") solution = nesterov(M, W, parameter, type, initialization, threshold, max_iter, verbose)
    if(acc_method == "anderson") solution = anderson(M, W, parameter, type, initialization, depth = acc_parameters$depth, threshold, max_iter, delay = acc_parameters$delay, guarded = acc_parameters$guarded, verbose)
    if(acc_method == "randerson") solution = randerson(M, W, parameter, type, initialization, depth = acc_parameters$depth, reg = list(reg_depth = acc_parameters$reg_depth, gamma = acc_parameters$gamma), threshold, max_iter, delay = acc_parameters$delay, guarded = acc_parameters$guarded, verbose)
  }
  if(method == "als"){
    if(is.null(initialization)){
      if(type == "hard") initialization = list(A = matrix(rnorm(n * parameter), n, parameter), B = matrix(rnorm(p * parameter), p, parameter))
      if(type == "soft") initialization = list(A = matrix(rnorm(n * p), n, p), B = matrix(rnorm(p * p), p, p))
    }
    if(acc_method == "baseline") solution = baseline_als(M, W, parameter, type, initialization, threshold, max_iter, verbose)
    if(acc_method == "nesterov") solution = nesterov_als(M, W, parameter, type, initialization, threshold, max_iter, verbose)
    if(acc_method == "anderson") solution = anderson_als(M, W, parameter, type, initialization, depth = acc_parameters$depth, threshold, max_iter, delay = acc_parameters$delay, guarded = acc_parameters$guarded, verbose)
    if(acc_method == "randerson") solution = randerson_als(M, W, parameter, type, initialization, depth = acc_parameters$depth, reg = list(reg_depth = acc_parameters$reg_depth, gamma = acc_parameters$gamma), threshold, max_iter, delay = acc_parameters$delay, guarded = acc_parameters$guarded, verbose)
  }
  return(solution)
}

############ PGD-based ############

loss = function(X, M, W, param, type){
  loss = 1/2 * sum(W * ((M - X)^2))
  if(type == "soft")  loss = loss + param * npmr::nuclear(X)
  return(loss/length(M))
}

gradient = function(X, M, W){
  return(W * M + (1 - W) * X)
}

project = function(X, param, type){
  SVD = svd(X)
  d = SVD$d
  if(type == "soft"){ 
    d = pmax(d - param, 0)
    rk = sum(d > 1e-8)
  }
  if(type == "hard") rk = param
  mat = SVD$u[,1:rk] %*% diag(d[1:rk]) %*% t(SVD$v[,1:rk])
  return(list(mat = mat, rk = rk))
}

############ Baseline ############

baseline = function(M, W, param, type, init, eps = 1e-8, max_iter = 100, verbose = FALSE){
  X0 = init$X
  X = X0
  obj = loss(X, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss(X, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    X0 = X
    obj0 = obj
    
    #update
    tictoc::tic()
    Y = gradient(X, M, W)
    proj = project(Y, param, type)
    X = proj$mat
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    obj = loss(X, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = sum((X - X0)^2)/sum(X0^2)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss(X, M, W, param, "hard"), proj$rk, delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  solution = list(X = X)
  return(list(solution = solution, info = info))
}

############ Nesterov ############

nesterov = function(M, W, param, type, init, eps = 1e-8, max_iter = 100, verbose = FALSE){
  X0 = init$X
  X = X0
  obj = loss(X, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss(X, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    obj0 = obj
    
    tictoc::tic()
    #nesterov
    X_new = nupdate(X, X0, iter)
    X0 = X
    X = X_new
    
    #update
    Y = gradient(X, M, W)
    proj = project(Y, param, type)
    X = proj$mat
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    obj = loss(X, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = sum((X - X0)^2)/sum(X0^2)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss(X, M, W, param, "hard"), proj$rk, delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  solution = list(X = X)
  return(list(solution = solution, info = info))
}


############ Anderson ############

anderson = function(M, W, param, type, init, depth = 3, eps = 1e-8, max_iter = 100, delay = 0, guarded = FALSE, verbose = FALSE){
  X0 = init$X
  X = X0
  Y0 = matrix(0, dim(X))
  sol = list(R = c(), F = c(), Sigma = NULL, alphas = c())
  obj = loss(X, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss(X, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    X0 = X
    obj0 = obj
    
    tictoc::tic()
    Y = gradient(X, M, W)
    if(iter > delay){
      upd = do.call(aupdate, c(sol, list(mat = Y, mat0 = Y0, depth = depth)))
      Y0 = Y
      sol = upd$sol
      Y = upd$mat
      proj = project(Y, param, type)
      X = proj$mat
      if(guarded & loss(X, M, W, param, type) > obj){
        if(verbose) cat('\nSkip Anderson\n')
        Y = Y0
        proj = project(Y, param, type)
        X = proj$mat
      }
    } else {
      proj = project(Y, param, type)
      X = proj$mat
    }
    Y0 = Y
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    obj = loss(X, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = sum((X - X0)^2)/sum(X0^2)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss(X, M, W, param, "hard"), proj$rk, delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  
  alphas = sol$alphas
  colnames(alphas) = 1:ncol(alphas)
  if(verbose) matplot(alphas, type = 'l', ylab = expression(alpha), xlab = "iteration", lty = 1)
  solution = list(X = X)
  return(list(solution = solution, info = info, coefs = alphas))
}

############ Anderson + regularization ############

randerson = function(M, W, param, type, init, depth = 3, reg = list(reg_depth = 3, gamma = 0), eps = 1e-8, max_iter = Inf, delay = 0, guarded = FALSE, verbose = FALSE){
  X0 = init$X
  X = X0
  Y0 = matrix(0, dim(X))
  sol = list(R = c(), F = c(), Sigma = NULL, alphas = c())
  obj = loss(X, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss(X, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    X0 = X
    obj0 = obj
    sol0 = sol
    
    tictoc::tic()
    Y = gradient(X, M, W)
    if(iter > delay){
      if(iter > delay + reg$reg_depth) upd = do.call(raupdate, c(sol, list(mat = Y, mat0 = Y0, depth = depth, reg = reg)))
      else upd = do.call(aupdate, c(sol, list(mat = Y, mat0 = Y0, depth = depth)))
      Y0 = Y
      sol = upd$sol
      Y = upd$mat
      proj = project(Y, param, type)
      X = proj$mat
      if(guarded & loss(X, M, W, param, type) > obj){
        if(verbose) cat('\nSkip Anderson\n')
        sol = sol0
        Y = Y0
        proj = project(Y, param, type)
        X = proj$mat
      }
    } else {
      proj = project(Y, param, type)
      X = proj$mat
    }
    Y0 = Y
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    obj = loss(X, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = sum((X - X0)^2)/sum(X0^2)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss(X, M, W, param, "hard"), proj$rk, delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  
  alphas = sol$alphas
  colnames(alphas) = 1:ncol(alphas)
  if(verbose) matplot(alphas, type = 'l', ylab = expression(alpha), xlab = "iteration", lty = 1)
  solution = list(X = X)
  return(list(solution = solution, info = info, coefs = alphas))
}

############ ALS-based ############

loss_als = function(A, B, M, W, param, type){
  loss = 1/2 * sum(W * ((M - A %*% t(B))^2))
  if(type == "soft") loss = loss + param/2 * (sum(A^2)+ sum(B^2))
  return(loss/length(M))
}

project_als = function(mat, X, param, type){
  S =  t(mat) %*% mat
  if(type == "soft") diag(S) = diag(S) + param
  t((solve(S) %*% t(mat)) %*% X)
}

step_als = function(A, B, X, W, param, type){
  Y = gradient(A %*% t(B), X, W)
  A = project_als(B, t(Y), param, type)
  Y = gradient(A %*% t(B), X, W)
  B = project_als(A, Y, param, type)
  return(list(A = A, B = B))
}

find_rank = function(A, B){
  SVD = svd(A)
  SVD = svd(B %*% SVD$v %*% diag(SVD$d))
  sum(SVD$d>1e-8)
}

inner = function(A0, B0, A, B) {
  sum(diag((t(A0) %*% A) %*% (t(B0) %*% B))) 
}

reldist = function(A0, B0, A, B){
  denom = inner(A0, B0, A0, B0)
  num = inner(A, B, A, B) - 2 * inner(A0, B0, A, B) + denom
  return(num/denom)
}

############ Baseline ############

baseline_als = function(M, W, param, type, init, eps = 1e-8, max_iter = 100, verbose = FALSE){
  A0 = init$A
  B0 = init$B
  A = A0
  B = B0
  X = A %*% t(B)
  obj = loss_als(A, B, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss_als(A, B, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    A0 = A
    B0 = B
    obj0 = obj
    
    #update
    tictoc::tic()
    upd = step_als(A, B, M, W, param, type)
    A = upd$A
    B = upd$B
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    obj = loss_als(A, B, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = reldist(A0, B0, A, B)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss_als(A, B, M, W, param, "hard"), find_rank(A, B), delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  solution = list(A = A, B = B)
  return(list(solution = solution, info = info))
}

############ Nesterov ############

nesterov_als = function(M, W, param, type, init, eps = 1e-8, max_iter = 100, verbose = FALSE){
  A0 = init$A
  B0 = init$B
  A = A0
  B = B0
  obj = loss_als(A, B, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss_als(A, B, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    obj0 = obj
    
    #update
    tictoc::tic()
    AB_new = nupdate(rbind(A, B), rbind(A0, B0), iter)
    A0 = A
    B0 = B
    A = AB_new[1:nrow(A),]
    B = AB_new[-(1:nrow(A)),]
    
    upd = step_als(A, B, M, W, param, type)
    A = upd$A
    B = upd$B
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    X_hat = A %*% t(B)
    obj = loss_als(A, B, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = reldist(A0, B0, A, B)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss_als(A, B, M, W, param, "hard"), find_rank(A, B), delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  solution = list(A = A, B = B)
  return(list(solution = solution, info = info))
}

############ Anderson ############

anderson_als = function(M, W, param, type, init, depth = 3, eps = 1e-8, max_iter = 100, delay = 0, guarded = FALSE, verbose = FALSE){
  A0 = init$A
  B0 = init$B
  upd = step_als(A0, B0, M, W, param, type)
  A = upd$A
  B = upd$B
  sol = list(R = c(), F = c(), Sigma = NULL, alphas = c())
  obj = loss_als(A, B, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss_als(A, B, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    obj0 = obj
    A0 = A
    B0 = B
    
    tictoc::tic()
    upd = step_als(A, B, M, W, param, type)
    if(iter > delay){
      upd_aa = do.call(aupdate, c(sol, list(mat = rbind(upd$A, upd$B), mat0 = rbind(A, B), depth = depth)))
      sol = upd_aa$sol
      AB_new = upd_aa$mat
      A = AB_new[1:nrow(A),]
      B = AB_new[-(1:nrow(A)),]
      if(guarded & loss_als(A, B, M, W, param, type) > obj){
        if(verbose) cat('\nSkip Anderson\n')
        A = upd$A
        B = upd$B
      }
    } else {
      A = upd$A
      B = upd$B
    }
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    X_hat = A %*% t(B)
    obj = loss_als(A, B, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = reldist(A0, B0, A, B)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss_als(A, B, M, W, param, "hard"), find_rank(A, B), delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  
  alphas = sol$alphas
  colnames(alphas) = 1:ncol(alphas)
  if(verbose) matplot(alphas, type = 'l', ylab = expression(alpha), xlab = "iteration", lty = 1)
  solution = list(A = A, B = B)
  return(list(solution = solution, info = info, coefs = alphas))
}

############ Anderson + regularization ############

randerson_als = function(M, W, param, type, init, depth = 3, reg = list(reg_depth = 3, gamma = 0), eps = 1e-8, max_iter = 100, delay = 0, guarded = FALSE, verbose = FALSE){
  A0 = init$A
  B0 = init$B
  upd = step_als(A0, B0, M, W, param, type)
  A = upd$A
  B = upd$B
  sol = list(R = c(), F = c(), Sigma = NULL, alphas = c())
  obj = loss_als(A, B, M, W, param, type)
  iter = 0
  delta = Inf
  
  info = data.frame(iter, 0, obj, loss_als(A, B, M, W, param, "hard"), NA, delta, Inf)
  info_names = c('iter', 'time', 'loss', 'loss_no_penalty', 'rank', 'delta', 'deltaX')
  colnames(info) = info_names
  
  while(delta > eps & iter < max_iter){
    iter = iter + 1
    
    #save
    obj0 = obj
    A0 = A
    B0 = B
    
    tictoc::tic()
    upd = step_als(A, B, M, W, param, type)
    if(iter > delay){
      if(iter > delay + reg$reg_depth) upd_aa = do.call(raupdate, c(sol, list(mat = rbind(upd$A, upd$B), mat0 = rbind(A, B), depth = depth, reg = reg)))
      else upd_aa = do.call(aupdate, c(sol, list(mat = rbind(upd$A, upd$B), mat0 = rbind(A, B), depth = depth)))
      sol = upd_aa$sol
      AB_new = upd_aa$mat
      A = AB_new[1:nrow(A),]
      B = AB_new[-(1:nrow(A)),]
      if(guarded & loss_als(A, B, M, W, param, type) > obj){
        if(verbose) cat('\nSkip Anderson\n')
        A = upd$A
        B = upd$B
      }
    } else {
      A = upd$A
      B = upd$B
    }
    time = tictoc::toc(quiet = TRUE)
    
    #evaluate
    X_hat = A %*% t(B)
    obj = loss_als(A, B, M, W, param, type)
    delta = abs((obj - obj0)/obj0)
    deltaX = reldist(A0, B0, A, B)
    
    res = data.frame(iter, time$toc - time$tic, obj, loss_als(A, B, M, W, param, "hard"), find_rank(A, B), delta, deltaX)
    colnames(res) = info_names
    info = rbind(info, res)
    if(verbose) cat("\n", paste(info_names, ':', format(res, digits = 5)))
  }
  rownames(info) = NULL
  info$time = cumsum(info$time)
  if(verbose) plot(log(info$loss), type = 'o', pch = 16)
  
  alphas = sol$alphas
  colnames(alphas) = 1:ncol(alphas)
  if(verbose) matplot(alphas, type = 'l', ylab = expression(alpha), xlab = "iteration", lty = 1)
  solution = list(A = A, B = B)
  return(list(solution = solution, info = info, coefs = alphas))
}

############ Nesterov ############

nupdate = function(mat, mat0, iter){
  return(mat + (iter - 1)/(iter + 2) * (mat - mat0))
}

############ Anderson ############

aupdate = function(mat, mat0, R, F, Sigma, alphas, depth){
  f = c(mat)
  r = f - c(mat0)
  R = cbind(R, r)
  F = cbind(F, f)
  if(is.null(Sigma)) Sigma = t(R) %*% R
  else{
    sigma = t(r) %*% R
    Sigma = cbind(Sigma, sigma[-length(sigma)])
    Sigma = rbind(Sigma, sigma)
  } 
  theta = solve(Sigma, rep(1, ncol(R)))
  alpha = theta/sum(theta)
  alphas = rbind(alphas, c(alpha, rep(0, depth + 1 - length(alpha))))
  mat = matrix(F %*% alpha, dim(mat))
  if(ncol(R) > depth){
    R = R[, -1]
    F = F[, -1]
    Sigma = Sigma[-1,]
    Sigma = Sigma[,-1]
  }
  sol = list(R = R, F = F, Sigma = Sigma, alphas = alphas)
  return(list(mat = mat, sol = sol))
}

############ Anderson + regularization ############

raupdate = function(mat, mat0, R, F, Sigma, alphas, depth, reg){
  f = c(mat)
  r = f - c(mat0)
  R = cbind(R, r)
  F = cbind(F, f)
  if(is.null(Sigma)) Sigma = t(R) %*% R
  else{
    sigma = t(r) %*% R
    Sigma = cbind(Sigma, sigma[-length(sigma)])
    Sigma = rbind(Sigma, sigma)
  }
  alpha0 = alphas[max(nrow(alphas) - reg$reg_depth + 1, 1):nrow(alphas), , drop = FALSE]
  alpha0 = colSums(alpha0)/nrow(alpha0)
  theta = solve(Sigma + diag(reg$gamma, nrow(Sigma)), rep(1, ncol(R)))
  alpha = theta/sum(theta)
  K = solve(Sigma +  diag(reg$gamma, nrow(Sigma))) %*% (outer(alpha0, rep(1, length(alpha0))) - outer(rep(1, length(alpha0)), alpha0))
  alpha = (diag(nrow(K)) + reg$gamma * K) %*% alpha
  alphas = rbind(alphas, c(alpha, rep(0, depth + 1 - length(alpha))))
  mat = matrix(F %*% alpha, dim(mat))
  if(ncol(R) > depth){
    R = R[, -1]
    F = F[, -1]
    Sigma = Sigma[-1,]
    Sigma = Sigma[,-1]
  }
  sol = list(R = R, F = F, Sigma = Sigma, alphas = alphas)
  return(list(mat = mat, sol = sol))
}


