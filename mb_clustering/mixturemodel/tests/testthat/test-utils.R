
test_that("l'initialisation ne contient pas de NA", {
  set.seed(1)
  X <- readRDS(system.file("testdata/X.rds", package = "mixturemodel"))
  parameters<-initialization(X, K=2)
  expect_false(any(is.na(parameters)))
})

test_that("les densités ne contiennent pas de NA", {
  set.seed(1)
  X <- readRDS(system.file("testdata/X.rds", package = "mixturemodel"))
  K<-2
  parameters<-initialization(X, K, outlier=TRUE)
  gaussian_D<-gaussian_densities(X, parameters[-(K+1)])
  uniform_D<-uniform_densities(X, parameters[[K]]$uniform_params)
  D<-mixte_densities(X, parameters)
  expect_false(any(is.na(gaussian_D)))
  expect_false(any(is.na(uniform_D)))
  expect_false(any(is.na(D)))
})
