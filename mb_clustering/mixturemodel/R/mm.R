

#' Expectation step
#'
#' @param x a matrix
#' @param parameters a list
#' @param outlier a boolean
#'
#' @returns a matrix
#' @export
#'
e_step<- function(x, parameters, outlier=FALSE){

  n<-dim(x)[1]
  K<-length(parameters)
  Q<-matrix(nrow=n, ncol=K)
  if (outlier){
    D<-mixte_densities(x, parameters)
  }
  else{
    D<-gaussian_densities(x, parameters)
  }
  Q<-D/rowSums(D)
  return(Q)
}

#' likehood
#'
#' @param e a function
#' @param x a matrix
#' @param parameters a list
#' @param outlier a boolean
#'
#' @returns a number
#' @export
#'
likehood <- function(e, x, parameters, outlier=FALSE){

  if (outlier){
    D<-mixte_densities(x, parameters)
  }
  else{
    D<-gaussian_densities(x, parameters)
  }
  Q<-e(x, parameters, outlier)
  pk<-sapply(parameters, function(x) x[[1]])
  eps <- 1e-300 #On corrige les zéros de la densité de la loi uniforme
  log_D <- log(pmax(D, eps))
  log_pk <- log(pmax(pk, eps))
  return(sum(Q * (log_D + log_pk)))
}

#' BIC
#'
#' @param e a function
#' @param x a matrix
#' @param parameters a list
#' @param outlier a boolean
#'
#' @returns a number
#' @export
#'
bic<-function(e, x, parameters, outlier=FALSE){

  n=dim(x)[1]
  d=dim(x)[2]
  K <- length(parameters)
  if (outlier){
    nu <- K + K * d + K * d * (d + 1) / 2 + 2 * d
  }
  else{
    nu <- (K - 1) + K * d + K * d * (d + 1) / 2
  }
  likehood_value<-likehood(e, x, parameters, outlier)
  return (2*likehood_value-nu*log(n))
}


#' Maximization step
#'
#' @param x a matrix
#' @param Q a matrix
#' @param K number of cluster
#' @param outlier a boolean
#'
#' @returns a list
#' @export
#'
m_step <- function(x, Q, K, outlier=FALSE, min_pk = 1e-6, reg_covar = 1e-6){
  x<-as.matrix(x)
  n<-dim(x)[1]
  d<-dim(x)[2]
  parameters<-list()
  for(k in 1:K){
    nk<-sum(Q[, k])
    pk<-max(nk / n, min_pk)
    if (is.na(nk) || nk < 1e-6){
      # réinitialise muk
      muk <- as.numeric(x[sample(1:n, 1), ])
      sigmak <- diag(rep(1e-2, d))  # petite covariance
    } else {
      muk <- colSums(x * Q[, k]) / nk
      diff <- x - matrix(muk, n, d, byrow = TRUE)
      sigmak <- t(diff) %*% (diff * Q[, k]) / nk

      # régularisation (ajouter eps * I)
      sigmak <- sigmak + diag(rep(reg_covar, d))

      # si Sigma_k mal conditionnée : forcer valeurs propres mini
      ev <- eigen(sigmak, symmetric = TRUE)
      vals <- ev$values
      vecs <- ev$vectors
      min_eig <- 1e-8
      if(any(vals < min_eig)) {
        vals[vals < min_eig] <- min_eig
        sigmak <- vecs %*% diag(vals) %*% t(vecs)
      }
    }
    parameters[[k]]<-list(pk = pk, muk = muk, sigmak = sigmak)
  }
  if (outlier){
    nk<-sum(Q[, K+1])
    pk <- max(nk / n, 1e-3)
    uniform_params <- numeric(2*d)

    for(j in 1:d) {
      uniform_params[2*j - 1] <- min(x[, j])
      uniform_params[2*j]     <- max(x[, j])
    }
    parameters[[K+1]]<-list(pk = pk, uniform_params=uniform_params)
  }

  # renormalise les pk pour que sum(pk)=1
  pks <- sapply(parameters, function(p) p$pk)
  pks <- pks / sum(pks)
  for(k in 1:length(parameters)) {parameters[[k]]$pk <- pks[k]}
  return(parameters)
}


#' EM Algorithm
#'
#' @param e a function
#' @param m a function
#' @param x a matrix
#' @param K a number
#' @param initial_parameters a list
#' @param outlier a boolean
#' @param epsilon a number
#'
#' @returns a list
#' @export
#'
em<- function(e, m, x, K, initial_parameters, outlier=FALSE, epsilon=1e-6){
  parameters_list<-list(initial_parameters)
  likehood_list<-list(likehood(e, x, initial_parameters, outlier))
  k<-1
  while (TRUE){
    Q<-e(x, initial_parameters,outlier)
    parameters<-m(x, Q, K, outlier)
    parameters_list[[k+1]]<-parameters
    likehood_list[[k+1]]<-likehood(e, x, parameters, outlier)
    if (is.na(likehood_list[[k+1]]) || is.na(likehood_list[[k]])) {
      stop(
        "Erreur : Matrices de  variance covariance mal conditionnées.\n",
        "Cause probable : d peut-être trop grand."
      )
    }
    else{
    if (abs(likehood_list[[k+1]]-likehood_list[[k]])<epsilon){
      break
    }
    k<-k+1
    initial_parameters<-parameters
    }
  }
  return(list(parameters=parameters, likehood_values=likehood_list))
}


#' Multistart EM
#'
#' @param e a function
#' @param m a function
#' @param x a matrix
#' @param K  a number
#' @param steps a number
#' @param outlier a boolean
#'
#' @returns a list
#' @export
#'
multistart_em<- function(e, m, x, K, steps=20, outlier=FALSE){
  likehood_vector<-vector(length = steps)
  likehood_values_list<-list()
  parameters_list<-list()
  for (step in 1:steps){
    initial_parameters<-initialization(x, K, outlier)
    em_values<-em(e, m, x, K, initial_parameters, outlier)
    parameters<-em_values$parameters
    likehood_values<-em_values$likehood_values
    parameters_list[[step]]<-parameters
    likehood_vector[step]<-likehood_values[[length(likehood_values)]]
    likehood_values_list[[step]]<-likehood_values
  }
  index<-which.max(likehood_vector)
  return(list(parameters=parameters_list[[index]], likehood_values=likehood_values_list[[index]]))
}


#' Select K
#'
#' @param e a function
#' @param m a function
#' @param x a matrix
#' @param K_max a number
#'
#' @returns a vector
#' @export
#'
k_selection<- function(e, m, x, K_max=10){

  bic_vector<-vector(length = K_max)
  for (k in 1:K_max){
    initial_parameters<-initialization(x, k)
    em_values<-em(e, m, x, k, initial_parameters)
    parameters<-em_values$parameters
    bic_value<-bic(e, x, parameters)
    bic_vector[k]<-bic_value
  }
  return(bic_vector)
}

#' Prediction function
#'
#' @param Q a matrix
#' @param classes a factor vector
#'
#' @returns a vector
#' @export
#'
pred <- function(Q, classes=NULL){
  preds = apply(Q, 1, which.max)
  if (is.null(classes)){
    return(preds)
  }
  else{
    preds_factor <- factor(preds, levels = 1:length(classes), labels = classes)
    return(preds_factor)
  }
}

