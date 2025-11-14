

#' Constructor function
#'
#' @param data a matrix
#' @param K a number
#' @param initialization_steps a number
#'
#' @returns None
#' @export
#'
MixtureModel <- function(data, K, initialization_steps=20) {

  structure(
    list(
      data = data,
      K = K,
      initialization_steps = initialization_steps,
      outlier =FALSE,
      bic_values = NULL,
      optimal_K = NULL,
      params = NULL,
      proba = NULL,
      preds = NULL,
      likehood_values = NULL,
      bic = NULL
    ),
    class = "mixturemodel"
  )
}

#' Generic fit
#'
#' @param object a mixturemodel object
#'
#' @returns None
#' @export
#'
fit <- function(object) {

  UseMethod("fit")
}


#' Fit function
#'
#' @param object a mixturemodel object
#'
#' @returns a mixturemodel object
#' @export
#'
fit.mixturemodel <- function(object) {

  #K_selection
  bic_values <- k_selection(e_step, m_step, object$data, object$K)
  optimal_K <- which.max(bic_values)
  object$bic_values <- bic_values
  object$optimal_K <- optimal_K
  #MixtureModel without outlier
  mixturemodel_without_outlier_em_values <- multistart_em(e_step, m_step, object$data, object$optimal_K, object$initialization_steps, outlier=FALSE)
  mixturemodel_without_outlier_params<-mixturemodel_without_outlier_em_values$parameters
  mixturemodel_without_outlier_likehood_values<-mixturemodel_without_outlier_em_values$likehood_values
  mixturemodel_without_outlier_bic <- bic(e_step, object$data, mixturemodel_without_outlier_params, outlier=FALSE)
  #MixtureModel with outlier
  mixturemodel_with_outlier_em_values <- multistart_em(e_step, m_step, object$data, object$optimal_K, object$initialization_steps, outlier=TRUE)
  mixturemodel_with_outlier_params<-mixturemodel_with_outlier_em_values$parameters
  mixturemodel_with_outlier_likehood_values<-mixturemodel_with_outlier_em_values$likehood_values
  mixturemodel_with_outlier_bic <- bic(e_step, object$data, mixturemodel_with_outlier_params, outlier=TRUE)
  if (mixturemodel_without_outlier_bic>=mixturemodel_with_outlier_bic){
    object$outlier<-FALSE
    object$params<-mixturemodel_without_outlier_params
    object$likehood_values<-mixturemodel_without_outlier_likehood_values
    object$bic<-mixturemodel_without_outlier_bic
    object$proba<-e_step(object$data, object$params, object$outlier)
    object$preds<-pred(object$proba)
    cat("Model trained without outlier.\n")
  }
  else{
    object$outlier<-TRUE
    object$params<-mixturemodel_with_outlier_params
    object$likehood_values<-mixturemodel_with_outlier_likehood_values
    object$bic<-mixturemodel_with_outlier_bic
    object$proba<-e_step(object$data, object$params, object$outlier)
    object$preds<-pred(object$proba)
    cat("Model trained with outlier.\n")
  }
  return(object)
}

