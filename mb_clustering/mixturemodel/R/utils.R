
#' Gaussian densities
#'
#' @param x a matrix
#' @param parameters a list
#'
#' @returns a matrix
#' @export a matrix
#' @import mvtnorm
#'
gaussian_densities <- function(x, parameters){

  n<-dim(x)[1]
  K=length(parameters)
  D <- matrix(nrow=n, ncol=K)
  for (k in 1:K){
    parameter<-parameters[[k]]
    mean <- parameter[[2]]
    sigma <- parameter[[3]]
    D[, k]<- dmvnorm(x, mean, sigma)
  }
  return(D)
}


#' Uniform densities
#'
#' @param x a matrix
#' @param parameters a list
#'
#' @returns a vector
#' @export
#'
uniform_densities <- function(x, parameters){

  n<-nrow(x)
  d<-ncol(x)

  a <- parameters[seq(1, 2*d, by=2)]
  b <- parameters[seq(2, 2*d, by=2)]
  volume <- prod(b - a)
  dens <- rep(1/volume, n)
  for(j in 1:d) {
    dens <- dens * (x[, j] >= a[j] & x[, j] <= b[j])
  }
  return(dens)
}

#' Mixte densities
#'
#' @param x a matrix
#' @param parameters a list
#'
#' @returns a matrix
#' @export
#'
mixte_densities <- function(x, parameters){

  n<-dim(x)[1]
  K=length(parameters)
  D <- matrix(nrow=n, ncol=K)
  D[, 1:K-1]<-gaussian_densities(x, parameters[-K])
  D[, K]<-uniform_densities(x, parameters[[K]]$uniform_params)
  return(D)
}

#' Initialization
#'
#' @param x a matrix
#' @param K a number
#' @param outlier a boolean
#'
#' @returns a list
#' @export
#' @import MCMCpack
#'
initialization<-function(x, K, outlier=FALSE){

  parameters<-list()
  if (outlier){
    pk <- rdirichlet(1, rep(1, K+1))
  }
  else{
    pk <- rdirichlet(1, rep(1, K))
  }
  d <- dim(x)[2]
  nu <- d + 2
  S <- diag(d)
  for (k in 1: K){
    muk<-vector(length = d)
    muk <-t(x[sample(1:nrow(x), 1), ])
    sigmak<-matrix(nrow=d, ncol=d)
    sigmak<- rWishart(1, nu, S)[,,1]
    parameters[[k]] <- list(pk = pk[k], muk = muk, sigmak = sigmak)
  }
  if (outlier){
    uniform_params <- numeric(2*d)
    for(j in 1:d) {
      xj <- x[, j]
      minx <- min(xj)
      maxx <- max(xj)
      width <- (maxx - minx)*0.8
      aj <- runif(1, min = minx, max = maxx - width)
      bj <- aj + width
      uniform_params[2*j - 1] <- aj
      uniform_params[2*j] <- bj
    }
    parameters[[K+1]]<-list(pk = pk[K+1], uniform_params=uniform_params)
  }
  return(parameters)
}
