#!/usr/bin/env Rscript
# Collect MLM sweep results into a CSV.
#
# Usage:
#   Rscript scripts/collect_sweep_results.R
#   Rscript scripts/collect_sweep_results.R --sweep runs/other_sweep --out results/other.csv
suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(fs)
  library(stringr)
  library(argparse)
})
# =============================================================================
# Configuration
# =============================================================================
DEFAULT_SWEEP_DIR <- "runs/chembl36_small_mask_mlm_lr_sweep"
METRIC_KEYS <- c(
  "eval_loss",
  "eval_masked_accuracy",
  "eval_perplexity",
  "train_loss",
  "epoch",
  "num_parameters",
  "train_runtime",
  "eval_runtime"
)
FINAL_EVAL_KEYS <- c(
  "eval_loss",
  "eval_masked_accuracy",
  "eval_perplexity"
)
DIR_RE <- "^mask_(.+?)__mlm_([0-9p]+)__lr_([0-9e\\-+.]+)$"
# =============================================================================
# Helpers
# =============================================================================
parse_mlm_prob <- function(raw) {
  as.numeric(str_replace_all(raw, "p", "."))
}
parse_dir_name <- function(name) {
  m <- str_match(name, DIR_RE)
  if (is.na(m[1, 1])) {
    return(NULL)
  }
  tibble(
    strategy = m[1, 2],
    mlm_prob = parse_mlm_prob(m[1, 3]),
    lr_from_name = m[1, 4]
  )
}
load_json_safe <- function(path) {
  if (!file_exists(path)) {
    return(list())
  }
  fromJSON(path, simplifyVector = FALSE)
}
is_numeric_scalar <- function(x) {
  is.numeric(x) && length(x) == 1 && !is.na(x)
}
as_scalar <- function(x, default = "") {
  if (is.null(x)) {
    return(default)
  }
  if (length(x) == 0) {
    return(default)
  }
  x[[1]]
}
as_numeric_or_blank <- function(x) {
  if (is.null(x)) {
    return("")
  }
  if (is.logical(x)) {
    return("")
  }
  if (is.numeric(x) && length(x) == 1) {
    return(as.numeric(x))
  }
  ""
}
get_named_value <- function(x, key, default = "") {
  if (is.null(x[[key]])) {
    return(default)
  }
  as_scalar(x[[key]], default = default)
}
eval_history <- function(trainer_state) {
  history <- trainer_state[["log_history"]]
  if (is.null(history) || !is.list(history)) {
    return(tibble())
  }
  history_tbl <- tibble(entry = history) |>
    mutate(
      eval_loss = map_dbl(
        entry,
        ~ {
          val <- .x[["eval_loss"]]
          if (is.numeric(val) && length(val) == 1 && !is.na(val)) {
            as.numeric(val)
          } else {
            NA_real_
          }
        }
      )
    ) |>
    filter(!is.na(eval_loss))
  if (nrow(history_tbl) == 0) {
    return(tibble())
  }
  history_tbl |>
    mutate(
      step = map(entry, ~ get_named_value(.x, "step", "")),
      eval_masked_accuracy = map(entry, ~ get_named_value(.x, "eval_masked_accuracy", ""))
    ) |>
    select(entry, step, eval_loss, eval_masked_accuracy)
}
best_logged_eval <- function(history_tbl) {
  if (nrow(history_tbl) == 0) {
    return(list())
  }
  history_tbl |>
    slice_min(eval_loss, n = 1, with_ties = FALSE) |>
    pull(entry) |>
    pluck(1)
}
last_logged_eval <- function(history_tbl) {
  if (nrow(history_tbl) == 0) {
    return(list())
  }
  history_tbl |>
    slice_tail(n = 1) |>
    pull(entry) |>
    pluck(1)
}
checkpoint_step <- function(checkpoint) {
  if (is.null(checkpoint) || !is.character(checkpoint) || !str_detect(checkpoint, "-")) {
    return("")
  }
  raw_step <- str_split(checkpoint, "-", simplify = TRUE)
  raw_step <- raw_step[1, ncol(raw_step)]
  parsed <- suppressWarnings(as.integer(raw_step))
  if (!is.na(parsed)) {
    parsed
  } else {
    raw_step
  }
}
collect_one_run <- function(run_dir) {
  run_name <- path_file(run_dir)
  parsed <- parse_dir_name(run_name)
  if (is.null(parsed)) {
    return(NULL)
  }
  results_path <- path(run_dir, "all_results.json")
  args_path <- path(run_dir, "run_args.json")
  if (!file_exists(results_path)) {
    message("  skip ", run_name, ": no all_results.json")
    return(NULL)
  }
  results <- load_json_safe(results_path)
  args <- load_json_safe(args_path)
  trainer_state <- load_json_safe(path(run_dir, "trainer_state.json"))
  history_tbl <- eval_history(trainer_state)
  best_logged <- best_logged_eval(history_tbl)
  last_logged <- last_logged_eval(history_tbl)
  best_checkpoint <- get_named_value(trainer_state, "best_model_checkpoint", "")
  load_best_model_at_end <- get_named_value(args, "load_best_model_at_end", "")
  metric_source <- if (!identical(load_best_model_at_end, FALSE) && !identical(best_checkpoint, "")) {
    "best_model_final_eval"
  } else {
    "final_model_eval"
  }
  best_step <- get_named_value(trainer_state, "best_global_step", "")
  if (identical(best_step, "")) {
    best_step <- checkpoint_step(best_checkpoint)
  }
  row <- tibble(
    run = run_name,
    size = get_named_value(args, "model_size", ""),
    strategy = parsed$strategy,
    mlm_prob = parsed$mlm_prob,
    learning_rate = get_named_value(args, "learning_rate", parsed$lr_from_name),
    eval_masking_strategy = get_named_value(args, "masking_strategy", parsed$strategy),
    eval_mlm_probability = get_named_value(args, "mlm_probability", parsed$mlm_prob),
    load_best_model_at_end = load_best_model_at_end,
    metric_source = metric_source,
    metric_note = paste(
      "Final eval after training; if load_best_model_at_end was enabled,",
      "the best checkpoint was loaded first. Validation masking uses this",
      "run-specific masking_strategy and mlm_probability."
    ),
    best_checkpoint = best_checkpoint,
    best_step = best_step,
    best_logged_eval_loss = get_named_value(best_logged, "eval_loss", ""),
    best_logged_eval_masked_accuracy = get_named_value(best_logged, "eval_masked_accuracy", ""),
    last_logged_eval_step = get_named_value(last_logged, "step", ""),
    last_logged_eval_loss = get_named_value(last_logged, "eval_loss", ""),
    last_logged_eval_masked_accuracy = get_named_value(last_logged, "eval_masked_accuracy", "")
  )
  final_metrics <- FINAL_EVAL_KEYS |>
    set_names(paste0("final_", FINAL_EVAL_KEYS)) |>
    map(~ get_named_value(results, .x, "")) |>
    as_tibble()
  legacy_metrics <- METRIC_KEYS |>
    set_names(METRIC_KEYS) |>
    map(~ get_named_value(results, .x, "")) |>
    as_tibble()
  bind_cols(row, final_metrics, legacy_metrics)
}
collect <- function(sweep_dir) {
  run_dirs <- dir_ls(sweep_dir, type = "directory") |>
    sort()
  rows <- map(run_dirs, collect_one_run) |>
    compact()
  if (length(rows) == 0) {
    return(tibble())
  }
  bind_rows(rows)
}
# =============================================================================
# CLI
# =============================================================================
parser <- ArgumentParser()
parser$add_argument(
  "--sweep",
  default = DEFAULT_SWEEP_DIR,
  help = "Sweep directory containing run subdirectories."
)
parser$add_argument(
  "--out",
  default = NULL,
  help = "Output CSV path. Defaults to <sweep>/sweep_results.csv."
)
args <- parser$parse_args()
sweep_dir <- args$sweep
out_file <- if (!is.null(args$out)) {
  args$out
} else {
  path(sweep_dir, "sweep_results.csv")
}
rows <- collect(sweep_dir)
if (nrow(rows) == 0) {
  message("No results found.")
  quit(status = 1)
}
write_csv(rows, out_file)
message("Wrote ", nrow(rows), " rows to ", out_file)

A shorter non-CLI version for interactive use:

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(fs)
  library(stringr)
})
sweep_dir <- "runs/chembl36_small_mask_mlm_lr_sweep"
out_file <- path(sweep_dir, "sweep_results.csv")
metric_keys <- c(
  "eval_loss",
  "eval_masked_accuracy",
  "eval_perplexity",
  "train_loss",
  "epoch",
  "num_parameters",
  "train_runtime",
  "eval_runtime"
)
final_eval_keys <- c(
  "eval_loss",
  "eval_masked_accuracy",
  "eval_perplexity"
)
parse_dir_name <- function(name) {
  m <- str_match(
    name,
    "^mask_(.+?)__mlm_([0-9p]+)__lr_([0-9e\\-+.]+)$"
  )
  if (is.na(m[1, 1])) {
    return(NULL)
  }
  tibble(
    strategy = m[1, 2],
    mlm_prob = as.numeric(str_replace_all(m[1, 3], "p", ".")),
    lr_from_name = m[1, 4]
  )
}
load_json_safe <- function(path) {
  if (file_exists(path)) {
    fromJSON(path, simplifyVector = FALSE)
  } else {
    list()
  }
}
get_value <- function(x, key, default = "") {
  if (is.null(x[[key]])) {
    default
  } else {
    x[[key]]
  }
}
get_eval_history <- function(trainer_state) {
  history <- trainer_state$log_history
  if (is.null(history) || !is.list(history)) {
    return(tibble(entry = list()))
  }
  tibble(entry = history) |>
    mutate(
      eval_loss = map_dbl(
        entry,
        ~ {
          val <- .x$eval_loss
          if (is.numeric(val) && length(val) == 1) val else NA_real_
        }
      )
    ) |>
    filter(!is.na(eval_loss))
}
checkpoint_step <- function(checkpoint) {
  if (is.null(checkpoint) || !is.character(checkpoint) || !str_detect(checkpoint, "-")) {
    return("")
  }
  raw_step <- str_split(checkpoint, "-", simplify = TRUE)
  raw_step <- raw_step[1, ncol(raw_step)]
  parsed <- suppressWarnings(as.integer(raw_step))
  ifelse(is.na(parsed), raw_step, parsed)
}
rows <- dir_ls(sweep_dir, type = "directory") |>
  sort() |>
  map(function(run_dir) {
    run_name <- path_file(run_dir)
    parsed <- parse_dir_name(run_name)
    if (is.null(parsed)) {
      return(NULL)
    }
    results_path <- path(run_dir, "all_results.json")
    if (!file_exists(results_path)) {
      message("  skip ", run_name, ": no all_results.json")
      return(NULL)
    }
    results <- load_json_safe(results_path)
    args <- load_json_safe(path(run_dir, "run_args.json"))
    trainer_state <- load_json_safe(path(run_dir, "trainer_state.json"))
    eval_history <- get_eval_history(trainer_state)
    best_logged <- if (nrow(eval_history) > 0) {
      eval_history |> slice_min(eval_loss, n = 1, with_ties = FALSE) |> pull(entry) |> pluck(1)
    } else {
      list()
    }
    last_logged <- if (nrow(eval_history) > 0) {
      eval_history |> slice_tail(n = 1) |> pull(entry) |> pluck(1)
    } else {
      list()
    }
    best_checkpoint <- get_value(trainer_state, "best_model_checkpoint", "")
    load_best_model_at_end <- get_value(args, "load_best_model_at_end", "")
    metric_source <- if (!identical(load_best_model_at_end, FALSE) && best_checkpoint != "") {
      "best_model_final_eval"
    } else {
      "final_model_eval"
    }
    best_step <- get_value(trainer_state, "best_global_step", "")
    if (best_step == "") {
      best_step <- checkpoint_step(best_checkpoint)
    }
    base_row <- tibble(
      run = run_name,
      size = get_value(args, "model_size", ""),
      strategy = parsed$strategy,
      mlm_prob = parsed$mlm_prob,
      learning_rate = get_value(args, "learning_rate", parsed$lr_from_name),
      eval_masking_strategy = get_value(args, "masking_strategy", parsed$strategy),
      eval_mlm_probability = get_value(args, "mlm_probability", parsed$mlm_prob),
      load_best_model_at_end = load_best_model_at_end,
      metric_source = metric_source,
      metric_note = paste(
        "Final eval after training; if load_best_model_at_end was enabled,",
        "the best checkpoint was loaded first. Validation masking uses this",
        "run-specific masking_strategy and mlm_probability."
      ),
      best_checkpoint = best_checkpoint,
      best_step = best_step,
      best_logged_eval_loss = get_value(best_logged, "eval_loss", ""),
      best_logged_eval_masked_accuracy = get_value(best_logged, "eval_masked_accuracy", ""),
      last_logged_eval_step = get_value(last_logged, "step", ""),
      last_logged_eval_loss = get_value(last_logged, "eval_loss", ""),
      last_logged_eval_masked_accuracy = get_value(last_logged, "eval_masked_accuracy", "")
    )
    final_metrics <- final_eval_keys |>
      set_names(paste0("final_", final_eval_keys)) |>
      map(~ get_value(results, .x, "")) |>
      as_tibble()
    legacy_metrics <- metric_keys |>
      set_names(metric_keys) |>
      map(~ get_value(results, .x, "")) |>
      as_tibble()
    bind_cols(base_row, final_metrics, legacy_metrics)
  }) |>
  compact() |>
  bind_rows()
if (nrow(rows) == 0) {
  stop("No results found.")
}
write_csv(rows, out_file)
message("Wrote ", nrow(rows), " rows to ", out_file)
