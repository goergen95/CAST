#' Calculate the risk-coverage curve of a selective prediction model
#' @description
#' The risk-coverage curve describes the behavior of a model that is combined with
#' a confidence score function used to select samples to exclude from prediction.
#' Compared to traditional reporting of risk at full coverage (1.0), the
#' risk-coverage curve describes the trade-off behavior of a selective prediction
#' model across different levels of coverage and their associated risks.
#'
#' The Area under the Risk-Coverage Curve (AURC) describes this trade-off
#' as a single scalar value. To make different models more comparable, the
#' excess-AURC (E-AURC) describes the excess area of an empirical risk-coverage
#' curve compared to a reference curve based on an optimal score function.
#' More information are presented in the details section and can be found in the linked
#' reference.
#' @author Darius A. Görgen
#' @details Note, that if \code{x} is a \code{data.frame}, it is required to
#' contain the columns \code{'score'} and \code{'residuals'}. In case \code{loss}
#' is specified, instead of the \code{'residuals'} column, the expected columns
#' are \code{'pred'} and \code{'obs'}. These columns are automatically retrieved
#' in the methods for other classes, or can be supplied manually.
#'
#' Given a fitted model \eqn{f} and a selection function \eqn{g}, the
#' combined selective prediction model is defined as:
#'
#' \deqn{
#'   (f,g)(x) \equiv \begin{cases}
#'   \text{$f(x)$, if $g(x) = 1$;}\\
#'   \text{reject, if $g(x) = 0$.}
#'   \end{cases}
#' }
#'
#' The empirical coverage of such a model over a given data set
#' \eqn{S_m = \{x_i,y_i\}_{i=1}^m} is defined as
#'
#' \deqn{
#'   \hat{\phi}(f,g|S_m) = \frac{1}{m} \sum_{i=1}^{m}g(x_i).
#' }
#'
#' While we leverage an arbitrary loss function \eqn{\mathcal{L}} to derive its
#' empirical selective risk
#'
#' \deqn{
#'   \hat r_s(f|S_m) = \frac{\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(f(x_i), y_i)g(x_i)}{\hat{\phi}(f,g|S_m)},
#' }
#' which serves as an estimator for the conditional probability \eqn{P(f(x) \ne y|g(x)=1)},
#' e.g. the risk of an undetected silent failure.
#'
#' The empirical generalized risk is defined as
#'
#' \deqn{
#'  \hat r_g(f|S_m) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(f(x_i), y_i)g(x_i),
#' }
#' which estimates the joint probability of prediction failure and sample
#' acceptance \eqn{P(f(x) \ne y, g(x)=1)}.
#'
#' The risk-coverage curve then describes the performance profile of such a selective
#' prediction model, defining risk as a function of coverage. The curve can be
#' derived for any confidence score function \eqn{\kappa} inducing a selection
#' function \eqn{g} by applying a threshold \eqn{\lambda}:
#'
#' \deqn{
#'   g_{\lambda}(x|\kappa,f) = \mathbb{1}[\kappa(x|f)>\lambda].
#' }
#'
#' @references Geifman, Y., Uziel, G., El-Yaniv, R., 2019. Bias-Reduced Uncertainty
#'   Estimation for Deep Neural Classifiers. https://doi.org/10.48550/arXiv.1805.08206
#'
#'   Traub, J., Bungert, T.J., Lüth, C.T., Baumgartner, M., Maier-Hein, K.H., Maier-Hein,
#'   L., Jaeger, P.F., 2024. Overcoming Common Flaws in the Evaluation of Selective
#'   Classification Systems. https://doi.org/10.48550/arXiv.2407.01032
#'
#' @param x A supported object for which an \code{rc} method is available. See
#'   more information in the details section.
#' @param ... Arguments passed to the data.frame method. There, it is forwarded to
#'   the loss-function, if specified. Otherwise ignored.
#' @returns A \code{\link{data.frame}} of class \code{rc} containing the columns
#'   \code{coverage}, \code{empirical}, \code{reference}, and \code{excess}. The
#'   estimates of the respective AUC values can be accessed via \code{attr(rc, "aurc")}.
#' @examples
#' library(CAST)
#'
#' set.seed(42)
#' obs <- runif(100)
#' pred <- obs + (runif(n = 100, max = 10) * obs)
#' residuals <- abs(pred - obs) # MAE
#' score <- runif(n = 100) * residuals # decreasing confidence with residuals
#'
#' # vectorized version with pre-calculated residuals
#' rc1 <- rc_vec(score = score, residuals = residuals)
#' # data.frame with pre-calculated residuals
#' rc2 <- rc(data.frame(score = score, residuals = residuals))
#' # data.frame with a specified loss function
#' rc3 <- rc(data.frame(pred = pred, obs = obs, score = score), loss = caret::MAE)
#'
#' identical(rc1, rc2) & identical(rc1, rc3)
#'
#' print(rc1)
#' attr(rc1, "auc")
#' plot(rc1)
#' @export
rc <- function(x, ...) UseMethod("rc")

#' @param score A numeric vector of confidence scores. Lower values represent higher confidence.
#' @param residuals A numeric vector of residuals. Lower values represent smaller errors.
#' @param risk A character indicating the type of risk to calculate. Must be one of
#'   \code{'generalized'} or \code{'selective'}.
#' @param n_bins An integer indicating the number of coverage bins between \code{[0,1]}.
#' @export
#' @name rc
#' @order 1
rc_vec <- function(
  score,
  residuals,
  risk = c("generalized", "selective"),
  n_bins = 100L
) {
  rc_impl(score, residuals, risk, n_bins)
}

#' @param loss By default, \code{NULL}. Can be set to a function. In this case,
#'   it is expected that the function receives at least the arguments \code{pred}
#'   and \code{obs} and returns a scalar. Additional arguments are forwarded using
#'   \code{...}.
#' @name rc
#' @export
rc.data.frame <- function(
  x,
  risk = c("generalized", "selective"),
  loss = NULL,
  n_bins = 100L,
  ...
) {
  stopifnot("score" %in% names(x))
  if (is.null(loss)) {
    if (!"residuals" %in% names(x)) {
      stop(
        "Either specify a loss function or make sure a `residuals` column is present."
      )
    }
  } else {
    stopifnot(c("pred", "obs") %in% names(x))
    stopifnot(inherits(loss, "function"))
    if (!all(c("pred", "obs") %in% names(formals(loss)))) {
      stop(
        "A custom loss function is required to accept arguments 'pred' and 'obs'."
      )
    }
    x["residuals"] <- mapply(
      loss,
      pred = x[["pred"]],
      obs = x[["obs"]],
      MoreArgs = list(...)
    )
  }
  rc <- rc_impl(x[["score"]], x[["residuals"]], risk, n_bins)
  return(rc)
}

#' @param aoa An (optional) object of type \code{\link{aoa}}. If not specified,
#'   \code{score} is required.
#' @name rc
#' @export
rc.train <- function(x, aoa, score, ...) {
  if (is.null(x[["pred"]])) {
    stop(paste(
      "Could not find predicted values in model object.",
      "\n  Did you forget to set `trControl = trainControl(savePredictions=TRUE)`?"
    ))
  }
  pred <- x[["pred"]]
  if (missing(aoa) && missing(score)) {
    stop(paste(
      "If supplying a 'train' object make sure to also supply either an aoa",
      "object or a vector of confidence scores."
    ))
  }
  if (!missing(aoa)) {
    score <- aoa[["parameters"]][["trainDI"]]
  }
  stopifnot(is.numeric(score))
  stopifnot(length(score) == nrow(pred))
  pred["score"] <- score
  return(rc(pred, ...))
}

#' @param model An (optional) object of type \code{\link[caret]{train}}. If not
#'   specified, \code{residuals} is required.
#' @name rc
#' @export
rc.aoa <- function(x, model, residuals, ...) {
  if (missing(model) && missing(residuals)) {
    stop(paste(
      "If supplying an 'aoa' object make sure to also supply either a",
      "`train` object or a vector of residuals."
    ))
  }
  if (!missing(model)) {
    return(rc(model, aoa = x, ...))
  }
  stopifnot(is.numeric(residuals))
  score <- x[["parameters"]][["trainDI"]]
  stopifnot(length(score) == length(residuals))
  pred <- data.frame(score = score, residuals = residuals)
  return(rc(pred, ...))
}

rc_impl <- function(
  score,
  residuals,
  risk = c("generalized", "selective"),
  n_bins = 100L
) {
  risk <- match.arg(risk)
  stopifnot(is.integer(n_bins), n_bins > 0)
  stopifnot(is.numeric(score) || is.numeric(residuals))
  stopifnot(all.equal(length(score), length(residuals)))

  rc <- .rc(score, residuals, risk)
  rc["reference"] <- .rc(residuals, residuals, risk)["risk"]
  if (nrow(rc) > n_bins) {
    rc <- .avg(rc, n_bins)
  }
  rc["excess"] <- rc["risk"] - rc["reference"]
  attr(rc, "auc") <- data.frame(
    risk = risk,
    auc_emp = .trapz(rc[["coverage"]], rc[["risk"]]),
    auc_ref = .trapz(rc[["coverage"]], rc[["reference"]]),
    auc_exs = .trapz(rc[["coverage"]], rc[["excess"]])
  )
  class(rc) <- c("rc", class(rc))
  return(unique(rc))
}

.rc <- function(score, residuals, risk) {
  order <- order(score, decreasing = TRUE)
  score <- score[order]
  residuals <- residuals[order]
  coverage <- cumsum(rep(1, length(score))) / length(score)
  risks <- (cumsum(residuals) / length(residuals))
  if (risk == "selective") {
    risks <- risks / coverage
  }
  rc <- data.frame(coverage = coverage, threshold = score, risk = risks)
  return(rc)
}

.avg <- function(rc, n_bins) {
  splits <- cut(rc[["coverage"]], breaks = seq(0, 1, 1 / n_bins))
  rc_split <- split(rc, splits, drop = FALSE)
  rc <- lapply(rc_split, sapply, max, na.rm = TRUE)
  rc <- as.data.frame(do.call(rbind, rc), row.names = NA)
  return(rc)
}

.trapz <- function(x, y) {
  sum((diff(x) / 2) * (y[1:(length(y) - 1)] + y[2:length(y)]))
}
