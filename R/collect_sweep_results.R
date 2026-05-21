#!/usr/bin/env Rscript
# Collect MLM sweep results into CSV(s).
#
# Outputs:
#   <sweep>/sweep_results.csv          — one row per run, all training metrics
#   <sweep>/fixed_eval_collected.csv   — fixed-eval per-prob results joined with
#                                        sweep metadata (written only if source exists)
#
# Usage:
#   Rscript R/collect_sweep_results.R
#   Rscript R/collect_sweep_results.R --sweep runs/other_sweep --out results/other.csv
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

DIR_RE <- "^mask_(.+?)__mlm_([0-9p]+)__lr_([0-9e\\-+.]+)$"

FINAL_EVAL_KEYS <- c("eval_loss", "eval_masked_accuracy", "eval_perplexity")

TRAINING_METRIC_KEYS <- c(
  "epoch",
  "train_loss",
  "num_parameters",
  "total_flos",
  "train_runtime",
  "train_samples_per_second",
  "train_steps_per_second",
  "train_samples_streaming",
  "eval_runtime",
  "eval_samples_per_second",
  "eval_steps_per_second"
)

RUN_ARG_KEYS <- c(
  "max_steps",
  "warmup_steps",
  "weight_decay",
  "per_device_train_batch_size",
  "gradient_accumulation_steps",
  "eval_steps",
  "span_p_geom",
  "span_max_length",
  "heteroatom_start_weight",
  "max_seq_length",
  "seed"
)

# =============================================================================
# Helpers
# =============================================================================
parse_mlm_prob <- function(raw) {
  as.numeric(str_replace_all(raw, "p", "."))
}

parse_dir_name <- function(name) {
  m <- str_match(name, DIR_RE)
  if (is.na(m[1, 1])) return(NULL)
  tibble(
    strategy    = m[1, 2],
    mlm_prob    = parse_mlm_prob(m[1, 3]),
    lr_from_name = m[1, 4]
  )
}

load_json_safe <- function(path) {
  if (!file_exists(path)) return(list())
  fromJSON(path, simplifyVector = FALSE)
}

as_scalar <- function(x, default = NA_character_) {
  if (is.null(x) || length(x) == 0) return(default)
  x[[1]]
}

get_chr <- function(x, key, default = NA_character_) {
  val <- x[[key]]
  if (is.null(val)) return(default)
  as.character(as_scalar(val, default))
}

get_dbl <- function(x, key, default = NA_real_) {
  val <- x[[key]]
  if (is.null(val) || is.logical(val)) return(default)
  parsed <- suppressWarnings(as.numeric(as_scalar(val, NA_character_)))
  if (is.na(parsed)) default else parsed
}

# =============================================================================
# Trainer state helpers
# =============================================================================
eval_history <- function(trainer_state) {
  history <- trainer_state[["log_history"]]
  if (is.null(history) || !is.list(history)) return(tibble())
  tibble(entry = history) |>
    mutate(
      eval_loss = map_dbl(entry, ~ {
        val <- .x[["eval_loss"]]
        if (is.numeric(val) && length(val) == 1 && !is.na(val)) as.numeric(val) else NA_real_
      })
    ) |>
    filter(!is.na(eval_loss))
}

best_logged_eval <- function(history_tbl) {
  if (nrow(history_tbl) == 0) return(list())
  history_tbl |> slice_min(eval_loss, n = 1, with_ties = FALSE) |> pull(entry) |> pluck(1)
}

last_logged_eval <- function(history_tbl) {
  if (nrow(history_tbl) == 0) return(list())
  history_tbl |> slice_tail(n = 1) |> pull(entry) |> pluck(1)
}

checkpoint_step <- function(checkpoint) {
  if (is.null(checkpoint) || !nzchar(checkpoint) || !str_detect(checkpoint, "-")) return(NA_integer_)
  parts <- str_split(checkpoint, "-", simplify = TRUE)
  parsed <- suppressWarnings(as.integer(parts[1, ncol(parts)]))
  if (!is.na(parsed)) parsed else NA_integer_
}

# =============================================================================
# Best-run metadata
# =============================================================================
load_best_run_names <- function(sweep_dir) {
  span     <- load_json_safe(path(sweep_dir, "best_span_run.json"))
  standard <- load_json_safe(path(sweep_dir, "best_standard_run.json"))
  list(
    span     = get_chr(span,     "run_name", NA_character_),
    standard = get_chr(standard, "run_name", NA_character_)
  )
}

# =============================================================================
# Per-run collection
# =============================================================================
collect_one_run <- function(run_dir) {
  run_name <- path_file(run_dir)
  parsed   <- parse_dir_name(run_name)
  if (is.null(parsed)) return(NULL)

  results_path <- path(run_dir, "all_results.json")
  if (!file_exists(results_path)) {
    message("  skip ", run_name, ": no all_results.json")
    return(NULL)
  }

  results      <- load_json_safe(results_path)
  args         <- load_json_safe(path(run_dir, "run_args.json"))
  trainer      <- load_json_safe(path(run_dir, "trainer_state.json"))
  history_tbl  <- eval_history(trainer)
  best_logged  <- best_logged_eval(history_tbl)
  last_logged  <- last_logged_eval(history_tbl)

  best_checkpoint       <- get_chr(trainer, "best_model_checkpoint", "")
  load_best_at_end      <- get_chr(args, "load_best_model_at_end", "")
  metric_source <- if (!identical(load_best_at_end, "FALSE") && nzchar(best_checkpoint)) {
    "best_model_final_eval"
  } else {
    "final_model_eval"
  }

  best_step <- get_dbl(trainer, "best_global_step", NA_real_)
  if (is.na(best_step)) best_step <- checkpoint_step(best_checkpoint)

  identity <- tibble(
    run              = run_name,
    size             = get_chr(args, "model_size", NA_character_),
    strategy         = parsed$strategy,
    mlm_prob         = parsed$mlm_prob,
    learning_rate    = get_dbl(args, "learning_rate", as.numeric(parsed$lr_from_name)),
    masking_strategy = get_chr(args, "masking_strategy", parsed$strategy),
    mlm_probability  = get_dbl(args, "mlm_probability",  parsed$mlm_prob)
  )

  checkpoint_info <- tibble(
    load_best_model_at_end = load_best_at_end,
    metric_source          = metric_source,
    best_checkpoint        = best_checkpoint,
    best_step              = best_step
  )

  logged <- tibble(
    best_logged_eval_loss             = get_dbl(best_logged, "eval_loss"),
    best_logged_eval_masked_accuracy  = get_dbl(best_logged, "eval_masked_accuracy"),
    last_logged_eval_step             = get_dbl(last_logged, "step"),
    last_logged_eval_loss             = get_dbl(last_logged, "eval_loss"),
    last_logged_eval_masked_accuracy  = get_dbl(last_logged, "eval_masked_accuracy")
  )

  final_metrics <- FINAL_EVAL_KEYS |>
    set_names(paste0("final_", FINAL_EVAL_KEYS)) |>
    map(~ get_dbl(results, .x)) |>
    as_tibble()

  training_metrics <- TRAINING_METRIC_KEYS |>
    map(~ get_dbl(results, .x)) |>
    set_names(TRAINING_METRIC_KEYS) |>
    as_tibble()

  run_args <- RUN_ARG_KEYS |>
    map(~ get_dbl(args, .x)) |>
    set_names(RUN_ARG_KEYS) |>
    as_tibble()

  bind_cols(identity, checkpoint_info, logged, final_metrics, training_metrics, run_args)
}

# =============================================================================
# Sweep-level collection
# =============================================================================
collect <- function(sweep_dir) {
  run_dirs <- dir_ls(sweep_dir, type = "directory") |> sort()
  rows <- map(run_dirs, collect_one_run) |> compact()

  if (length(rows) == 0) return(tibble())

  result <- bind_rows(rows)

  best <- load_best_run_names(sweep_dir)
  result |> mutate(
    is_best_span     = run == best$span,
    is_best_standard = run == best$standard
  )
}

collect_fixed_eval <- function(sweep_dir, sweep_results) {
  fixed_path <- path(sweep_dir, "fixed_eval_per_prob_results.csv")
  if (!file_exists(fixed_path)) return(NULL)

  fixed <- read_csv(fixed_path, show_col_types = FALSE)

  meta <- sweep_results |>
    select(run, size, is_best_span, is_best_standard) |>
    distinct()

  left_join(fixed, meta, by = c("run_name" = "run"))
}

# =============================================================================
# CLI
# =============================================================================
parser <- ArgumentParser()
parser$add_argument(
  "--sweep",
  default = DEFAULT_SWEEP_DIR,
  help    = "Sweep directory containing run subdirectories."
)
parser$add_argument(
  "--out",
  default = NULL,
  help    = "Output CSV path for sweep results. Defaults to <sweep>/sweep_results.csv."
)
parser$add_argument(
  "--fixed-eval-out",
  default = NULL,
  help    = paste(
    "Output CSV for fixed-eval results joined with sweep metadata.",
    "Defaults to <sweep>/fixed_eval_collected.csv.",
    "Skipped if fixed_eval_per_prob_results.csv is absent."
  )
)

args      <- parser$parse_args()
sweep_dir <- args$sweep
out_file  <- args$out %||% path(sweep_dir, "sweep_results.csv")
fixed_out <- args$fixed_eval_out %||% path(sweep_dir, "fixed_eval_collected.csv")

rows <- collect(sweep_dir)
if (nrow(rows) == 0) {
  message("No results found.")
  quit(status = 1)
}
write_csv(rows, out_file)
message("Wrote ", nrow(rows), " rows to ", out_file)

fixed <- collect_fixed_eval(sweep_dir, rows)
if (!is.null(fixed)) {
  write_csv(fixed, fixed_out)
  message("Wrote ", nrow(fixed), " rows to ", fixed_out)
} else {
  message("No fixed_eval_per_prob_results.csv found — skipping fixed eval output.")
}
