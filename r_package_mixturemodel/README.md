# mixturemodel

An R package for **model-based clustering** using **Gaussian Mixture Models (GMMs)** with integrated outlier detection.

## Overview

`mixturemodel` provides an implementation of Gaussian Mixture Models for unsupervised clustering. The package allows users to estimate mixture models, identify latent groups within data and detect potential outliers.

The project was developed as part of a Data Science portfolio and follows the standard structure of an R package.

---

## Features

* Gaussian Mixture Model (GMM) estimation
* EM algorithm for parameter estimation
* Outlier detection
* Cluster parameter estimation
* Package documentation and examples
* Unit tests

---

## Project Structure

```text
r_package_mixturemodel/                                
├── package/
│   └── mixturemodel/
│       ├── R/
│       ├── man/
│       ├── inst/
│       ├── tests/
│       ├── vignettes/
│       ├── DESCRIPTION
│       ├── NAMESPACE
│       ├── .Rbuildignore
│       ├── .gitignore
│       └── mixturemodel.Rproj
└── README.md
```

---

## Installation

Install the package from the source archive.

```r
install.packages(
  "mixturemodel_0.1.0.tar.gz",
  repos = NULL,
  type = "source"
)
```

---

## Quick Start

```r
library(mixturemodel)

model <- MixtureModel(
  X,
  K = 2,
  initialization_steps = 20
)

model <- fit(model)

summary(model)

tail(model$params)
```

---

## Documentation

A complete tutorial describing the package architecture, available functions, examples and unit tests is available in the package vignette.

---

## Technologies

* R
* Gaussian Mixture Models
* roxygen2
* devtools
* testthat

---

## Author

Developed by Thalia as part of a Data Science portfolio.

