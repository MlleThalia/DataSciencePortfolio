library(ggplot2)
library(e1071)

####################################################
###### Load and split train/val/test for univariate

load_and_split <- function(
    data, freq = 12,
    val_freq = 2,
    test_freq= 1,
    verbose = FALSE
    ) {
  
  # split train / val / test
  n <- length(data)
  
  train <- ts(data[1:(n-val_freq*freq)], frequency = freq)
  val <- ts(data[(n-val_freq*freq+1):(n-freq)], frequency = freq)
  train_final <- ts(data[1:(n-freq)], frequency = freq)
  test <- ts(data[(n-freq+1):n], frequency = freq)
  
  if (verbose) {
    cat("Train:", length(train), "obs (", length(train)/freq, "jours)\n")
    cat("Validation:", length(val), "obs (", length(val)/freq, "jours)\n")
    cat("Train all:", length(train_final), "obs (", length(train_final)/freq, "jours)\n")
    cat("Test:", length(test), "obs (", length(test)/freq, "jours)\n\n")
  }
  
  return(list(
    full = data,
    train = train,
    val = val,
    train_final = train_final,
    test = test
  ))
}

#######################################################
###### Function to train NNAR

train_nnar_fn <- function(ts_object, params, repeats=1, xreg=NULL){
  model <- nnetar(ts_object, p = params$p, P = params$P, size = params$size, repeats=repeats, xreg=xreg)
  return(model)
}

######################################################
##### Fit & prediction for NNAR
grid_search_nnar <- function(
    train_ts, val_ts, 
    hyperparam_grid, 
    repeats = 1, 
    xreg_train = NULL, 
    xreg_val = NULL
    ){
  
  results <- data.frame(
    p = numeric(0),
    P = numeric(0),
    size = numeric(0),
    rmse = numeric(0)
  )
  
  best_rmse <- Inf
  best_model <- NULL
  best_pred_val <- NULL
  best_hyperparams <- NULL
  
  for(i in 1:nrow(hyperparam_grid)){
    p <- hyperparam_grid$p[i]
    P <- hyperparam_grid$P[i]
    size <- hyperparam_grid$size[i]
    
    # fit NNAR
    model <- nnetar(train_ts, p = p, P = P, size = size, repeats = repeats, xreg = xreg_train)
    
    # predict validation data
    pred_val <- forecast(model, h = length(val_ts), xreg = xreg_val)$mean
    pred_val <- ts(pred_val, frequency = frequency(train_ts))
    
    # rmse
    rmse_val <- sqrt(mean((as.numeric(pred_val) - as.numeric(val_ts))^2, na.rm = TRUE))
    
    results <- rbind(results, data.frame(p = p, P = P, size = size, rmse = rmse_val))
    
    if(rmse_val < best_rmse){
      best_rmse <- rmse_val
      best_model <- model
      best_pred_val <- ts(pred_val, frequency = frequency(val_ts))
      best_hyperparams <- hyperparam_grid[i, ]
    }
  }
  
  return(list(
    best_model = best_model,
    best_pred_val = best_pred_val,
    best_rmse = best_rmse,
    best_hyperparams = best_hyperparams,
    results = results
  ))
}


#######################################################
###### Evaluate prediction
eval_pred <- function(pred_ts, true_ts, model_name, return_metrics = FALSE) {
  rmse <- sqrt(mean((as.numeric(pred_ts) - as.numeric(true_ts))^2, na.rm = TRUE))
  mae  <- mean(abs(as.numeric(pred_ts) - as.numeric(true_ts)), na.rm = TRUE)
  mape <- mean(abs((as.numeric(pred_ts) - as.numeric(true_ts)) / as.numeric(true_ts)), na.rm = TRUE) * 100
  
  cat(paste0("\n=== ", model_name, " - Evaluation on validation data ===\n"))
  cat("RMSE:", round(rmse, 2), "\n")
  cat("MAE :", round(mae, 2), "\n")
  cat("MAPE:", round(mape, 2), "%\n\n")
  
  if (return_metrics) {return(list(rmse = rmse, mae = mae, mape = mape))}
}

####################################################
###### Plot prediction of validation data

plot_pred_val <- function(
    train_ts, val_ts, pred_val_ts,
    zoom_months = 3,model_name
    ) {
  
  n_train <- length(train_ts)
  h <- length(val_ts)
  freq <- frequency(train_ts)
  
  # axe temporel de la validation
  time_val <- seq(time(train_ts)[n_train], by = 1/freq, length.out = h)
  
  # borne x pour le zoom
  x_min <- time(train_ts)[n_train - freq * zoom_months]
  x_max <- time_val[h]
  
  # borne y : inclure train, val, et prédictions
  ylim_range <- range(c(train_ts[(n_train - freq * zoom_months):n_train],
                        val_ts, pred_val_ts))
  
  # plot train
  plot(train_ts,
       xlim = c(x_min, x_max),
       ylim = ylim_range,
       main = paste(model_name, "- Prediction on validation data"),
       ylab = "ppm",
       xlab = "Temps")
  
  # plot validation 
  lines(time_val, val_ts, col = "red", lwd = 2)
  
  # plot prédictions
  lines(time_val, pred_val_ts, col = "blue", lwd = 2)
  
  # légende
  legend("topleft",
         legend = c("Train", "Validation", "Prediction"),
         col = c("black", "red", "blue"),
         lwd = 2)
}

#########################################################
############ Plot prediction of test data

plot_pred_test <- function(train_ts, pred_ts, zoom_months = 3, model_name) {
  n_train <- length(train_ts)
  h <- length(pred_ts)
  freq <- frequency(train_ts)
  
  # axe temporel de la prédiction
  time_test <- seq(time(train_ts)[n_train], 
                   by = 1/freq, 
                   length.out = h)
  
  # plot train
  plot(train_ts,
       xlim = c(time(train_ts)[n_train - freq*zoom_months], 
                time_test[h]),
       ylim = range(c(train_ts[(n_train - freq*zoom_months):n_train], pred_ts)),
       main = paste(model_name, "- Prediction on test data"),
       ylab = "ppm", xlab = "Temps")
  
  # plot prediction
  lines(time_test, pred_ts, col = "blue", lwd = 2)
  
  # légende
  legend("topleft", legend = c("Train+Val", "Prediction"), 
         col = c("black", "blue"), lwd = 2)
}

##########################################################
############# Save predictions to csv

save_predictions <- function(pred, filename, colname = "prediction") {

  df <- data.frame(pred)
  colnames(df) <- colname
  write.csv(df, filename, row.names = FALSE)
  
  cat("Predictions saved to", filename, "\n")
}