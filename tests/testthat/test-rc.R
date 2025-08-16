test_that("test rc.data.frame works as expected", {
  set.seed(152)
  obs <- runif(100)
  pred <- obs + (runif(n = 100, max = 10) * obs)
  residuals <- abs(pred - obs)
  score <- runif(n = 100) * obs
  data <- data.frame(
    obs = obs,
    pred = pred,
    residuals = residuals,
    score = score
  )
  # check rc_vec
  expect_error(rc_vec(score), "residuals")
  expect_error(rc_vec(residuals = residuals), "score")
  expect_error(rc_vec(score, residuals, risk = "other"), "should be one of")
  expect_silent(rc1 <- rc_vec(score, residuals, risk = "selective"))
  expect_s3_class(rc1, "rc")
  expect_equal(nrow(rc1), 100)
  expect_equal(attr(rc1, "auc")$risk, "selective")
  expect_silent(rc2 <- rc_vec(score, residuals, risk = "generalized"))
  expect_s3_class(rc2, "rc")
  expect_equal(nrow(rc2), 100)
  expect_equal(attr(rc2, "auc")$risk, "generalized")
  expect_false(identical(rc1$risk, rc2$risk))
  expect_silent(rc3 <- rc_vec(score, residuals, n_bins = 50L))
  expect_s3_class(rc3, "rc")
  expect_equal(nrow(rc3), 50)
  # check for correct behavior with different argument constellations
  expect_error(rc(score), "no applicable method")
  expect_error(rc(subset(data, select = -score)), "score")
  expect_error(rc(data, n_bins = 1.0), "integer")
  expect_error(rc(data, n_bins = -1L), "> 0")
  expect_error(rc(subset(data, select = -residuals)), "residuals")
  expect_error(rc(subset(data, select = -pred), loss = caret::MAE), "pred")
  expect_error(rc(data, loss = "a"), "function")
  expect_error(rc(data, loss = function(x) {}), "arguments")
  # check calculation works
  expect_silent(rc4 <- rc(subset(data, select = -c(pred, obs))))
  expect_silent(rc5 <- rc(subset(data, select = -residuals), loss = caret::MAE))
  expect_identical(rc4, rc5)
  expect_equal(
    names(rc4),
    c("coverage", "threshold", "risk", "reference", "excess")
  )
  expect_equal(nrow(rc4), 100)
  expect_s3_class(rc5, "rc")
  expect_silent(auc <- attr(rc4, "auc"))
  expect_equal(auc$risk, "generalized")
  expect_equal(
    round(auc[, 2:4], 4),
    data.frame(auc_emp = 1.5925, auc_ref = 1.8789, auc_exs = -0.2864),
    tolerance = 1e-4
  )
  expect_silent(
    rc6 <- rc(subset(data, select = -c(pred, obs)), risk = "selective")
  )
  expect_silent(auc <- attr(rc6, "auc"))
  expect_equal(auc$risk, "selective")
  expect_equal(
    round(auc[, 2:4], 4),
    data.frame(auc_emp = 3.3308, auc_ref = 4.6838, auc_exs = -1.3529)
  )
  # check specifying custom bins works
  expect_silent(rc7 <- rc(data, n_bins = 75L))
  expect_equal(nrow(rc7), 75)
  # check plot utilities with NA rows
  rc5[60, c(2:4)] <- NA
  expect_silent(gg <- plot(rc5))
  expect_s3_class(gg, "gg")
  expect_equal(
    gg$labels,
    list(
      x = "Coverage",
      y = "Risk",
      colour = "Type",
      title = "generalized risk"
    )
  )
  expect_equal(names(gg$data), c("coverage", "curve", "risk"))
  expect_equal(
    as.character(unique(gg$data$curve)),
    c("empirical", "reference", "excess")
  )
  expect_silent(gg <- plot(rc6, empirical = FALSE, reference = FALSE))
  expect_s3_class(gg, "gg")
  expect_equal(
    gg$labels,
    list(x = "Coverage", y = "Risk", colour = "Type", title = "selective risk")
  )
  expect_equal(names(gg$data), c("coverage", "curve", "risk"))
  expect_equal(as.character(unique(gg$data$curve)), "excess")
})

loaddata <- function() {
  # prepare sample data:
  data(cookfarm)
  dat <- aggregate(
    cookfarm[, c("VW", "Easting", "Northing")],
    by = list(as.character(cookfarm$SOURCEID)),
    mean
  )
  pts <- sf::st_as_sf(dat, coords = c("Easting", "Northing"))
  pts$ID <- 1:nrow(pts)
  set.seed(100)
  pts <- pts[1:30, ]
  studyArea <- terra::rast(system.file(
    "extdata",
    "predictors_2012-03-25.tif",
    package = "CAST"
  ))[[1:8]]
  trainDat <- terra::extract(studyArea, pts, na.rm = FALSE)
  trainDat <- merge(trainDat, pts, by.x = "ID", by.y = "ID")

  # train a model:
  set.seed(100)
  variables <- c("DEM", "NDRE.Sd", "TWI")
  ctrl <- caret::trainControl(method = "cv", number = 5, savePredictions = T)
  model <- caret::train(
    trainDat[, which(names(trainDat) %in% variables)],
    trainDat$VW,
    method = "rf",
    importance = TRUE,
    tuneLength = 1,
    trControl = ctrl
  )

  AOA <- aoa(studyArea, model, verbose = F)

  return(list(
    model = model,
    aoa = AOA
  ))
}

test_that("test rc methods work as expected", {
  skip_if_not_installed("randomForest")
  data <- loaddata()
  # model first
  expect_error(rc(data$model), "aoa")
  expect_error(rc(data$model, aoa = data$aoa), "loss")
  expect_error(rc(data$model, score = data$aoa$parameters$trainDI), "residuals")
  expect_error(
    rc(data$model, aoa = data$aoa, risk = "other", loss = caret::MAE),
    "should be one of"
  )
  expect_silent(rc1 <- rc(data$model, aoa = data$aoa, loss = caret::MAE))
  expect_silent(
    rc2 <- rc(
      data$model,
      score = data$aoa$parameters$trainDI,
      loss = caret::MAE
    )
  )
  expect_silent(rc(
    data$model,
    aoa = data$aoa,
    risk = "generalized",
    loss = caret::MAE
  ))
  model2 <- data$model
  model2$pred <- NULL
  expect_error(rc(model2, aoa = data$aoa, loss = caret::MAE), "savePredictions")
  # aoa first
  expect_error(rc(data$aoa), "aoa")
  expect_error(rc(data$aoa, model = data$model), "loss")
  expect_error(
    rc(data$aoa, model = data$model, risk = "other", loss = caret::MAE),
    "should be one of"
  )
  expect_silent(rc3 <- rc(data$aoa, model = data$model, loss = caret::MAE))
  residuals <- abs(data$model$pred$pred - data$model$pred$obs)
  expect_silent(rc4 <- rc(data$aoa, residuals = residuals))
  # compare
  expect_identical(rc1, rc2, rc3, rc4)
  expect_s3_class(rc1, "rc")
  expect_equal(nrow(rc1), 30)
  # argument forwarding
  expect_silent(
    rc5 <- rc(data$aoa, model = data$model, loss = caret::MAE, n_bins = 20L)
  )
  expect_equal(nrow(rc5), 20)
  # custom loss
  loss <- function(arg) {}
  expect_error(
    rc(data$aoa, model = data$model, loss = loss),
    "arguments 'pred' and 'obs'"
  )
  loss <- function(pred, obs) {
    abs(pred - obs)
  }
  expect_silent(rc6 <- rc(data$aoa, model = data$model, loss = loss))
  expect_identical(rc1, rc6)
  loss <- function(pred, obs, plus = 10) {
    abs(pred - obs) + plus
  }
  expect_silent(
    rc7 <- rc(data$aoa, model = data$model, loss = loss, plus = 100)
  )
  expect_true(rc7$risk[30] > 100)
})
