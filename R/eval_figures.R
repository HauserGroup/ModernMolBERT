#!/usr/bin/env Rscript

# rank_and_pairwise_model_benchmark.R
#
# Purpose:
#   Simple visualization of model performance across benchmark datasets.
#   Produces:
#     1. paired within-dataset rank dot plot
#     2. model rank summary table
#     3. compact pairwise win-count matrix
#
# Input:
#   best_metric_by_dataset_embedder.csv
#
# Required columns:
#   dataset
#   embedder
#   test_metric
#
# Usage:
#   Rscript rank_and_pairwise_model_benchmark.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(stringr)
})

# -------------------------------------------------------------------------
# User settings
# -------------------------------------------------------------------------

input_csv <- "outputs/best_metric_by_dataset_embedder.csv"
output_dir <- "outputs/dabestr"

# Replace the four ModernMolBERT names with the exact names in your CSV.
embedder_keep <- c(
  # comparison models
  "ECFP",
  "ECFP_count",
  "mol2vec",
  "MoLFormer-XL-both-10pct",
  "ChemBERTa-77M-MLM",
  "SELFormer",
  "CDDD",
  "CLAMP",

  # your models
  "ModernMolBERT-small",
  "ModernMolBERT-base",
  "ModernMolBERT-small-APE",
  "ModernMolBERT-base-APE"
)

# Difference below this threshold is treated as a tie in pairwise wins.
# 0.001 = 0.1 AUROC percentage points.
tie_threshold <- 0.001

# Whether to drop datasets that do not contain all selected embedders.
require_complete_dataset_coverage <- TRUE

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------------------------------------------------
# Load and validate data
# -------------------------------------------------------------------------

df <- read_csv(input_csv, show_col_types = FALSE)

required_cols <- c("dataset", "embedder", "test_metric")
missing_cols <- setdiff(required_cols, names(df))

if (length(missing_cols) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_cols, collapse = ", ")
  )
}

df <- df %>%
  mutate(
    dataset = as.character(dataset),
    embedder = as.character(embedder),
    test_metric = as.numeric(test_metric)
  )

available_embedders <- sort(unique(df$embedder))
missing_embedders <- setdiff(embedder_keep, available_embedders)

if (length(missing_embedders) > 0) {
  warning(
    "These selected embedders were not found in the CSV:\n  ",
    paste(missing_embedders, collapse = "\n  "),
    "\n\nAvailable embedders are:\n  ",
    paste(available_embedders, collapse = "\n  ")
  )
}

embedder_keep <- intersect(embedder_keep, available_embedders)

if (length(embedder_keep) < 2) {
  stop("Fewer than two selected embedders were found in the CSV.")
}

plot_df <- df %>%
  filter(embedder %in% embedder_keep) %>%
  select(dataset, embedder, test_metric) %>%
  filter(!is.na(test_metric))

# -------------------------------------------------------------------------
# Dataset coverage
# -------------------------------------------------------------------------

coverage <- plot_df %>%
  count(dataset, embedder, name = "n") %>%
  pivot_wider(
    names_from = embedder,
    values_from = n,
    values_fill = 0
  )

write_csv(
  coverage,
  file.path(output_dir, "dataset_embedder_coverage.csv")
)

if (require_complete_dataset_coverage) {
  complete_datasets <- plot_df %>%
    group_by(dataset) %>%
    summarise(
      n_embedders = n_distinct(embedder),
      .groups = "drop"
    ) %>%
    filter(n_embedders == length(embedder_keep)) %>%
    pull(dataset)

  plot_df <- plot_df %>%
    filter(dataset %in% complete_datasets)
}

if (nrow(plot_df) == 0) {
  stop(
    "No rows left after filtering. Check embedder names and dataset coverage."
  )
}

message("Using ", n_distinct(plot_df$dataset), " datasets.")
message("Using ", n_distinct(plot_df$embedder), " embedders.")

# -------------------------------------------------------------------------
# Within-dataset ranks
# -------------------------------------------------------------------------

rank_df <- plot_df %>%
  group_by(dataset) %>%
  mutate(
    rank = rank(-test_metric, ties.method = "average")
  ) %>%
  ungroup()

model_order <- rank_df %>%
  group_by(embedder) %>%
  summarise(
    mean_rank = mean(rank, na.rm = TRUE),
    mean_auc = mean(test_metric, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_rank, desc(mean_auc)) %>%
  pull(embedder)

rank_df <- rank_df %>%
  mutate(
    embedder = factor(embedder, levels = rev(model_order))
  )

write_csv(
  rank_df,
  file.path(output_dir, "ranked_dataset_embedder_long.csv")
)

# -------------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------------

n_models <- length(model_order)

rank_summary <- rank_df %>%
  group_by(embedder) %>%
  summarise(
    n_datasets = n(),
    mean_auc = mean(test_metric, na.rm = TRUE),
    median_auc = median(test_metric, na.rm = TRUE),
    sd_auc = sd(test_metric, na.rm = TRUE),
    mean_rank = mean(rank, na.rm = TRUE),
    median_rank = median(rank, na.rm = TRUE),
    sd_rank = sd(rank, na.rm = TRUE),
    n_top1 = sum(rank == 1, na.rm = TRUE),
    n_top3 = sum(rank <= 3, na.rm = TRUE),
    n_bottom3 = sum(rank >= n_models - 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_rank)

write_csv(
  rank_summary,
  file.path(output_dir, "model_rank_summary.csv")
)

print(rank_summary)

# -------------------------------------------------------------------------
# Figure 1: rank dot plot
# -------------------------------------------------------------------------

p_rank <- ggplot(rank_df, aes(x = rank, y = embedder)) +
  geom_point(
    alpha = 0.45,
    size = 1.8,
    position = position_jitter(height = 0.12, width = 0)
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 18,
    size = 3.8
  ) +
  scale_x_reverse(
    breaks = seq(1, n_models, by = 1),
    limits = c(n_models + 0.5, 0.5)
  ) +
  labs(
    x = "Within-dataset rank (1 = best)",
    y = NULL,
    title = "Model ranks across benchmark datasets",
    subtitle = "Points are datasets; diamonds show mean rank"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 10)
  )

ggsave(
  file.path(output_dir, "model_rank_dotplot.pdf"),
  p_rank,
  width = 7.5,
  height = 5.2
)

ggsave(
  file.path(output_dir, "model_rank_dotplot.png"),
  p_rank,
  width = 7.5,
  height = 5.2,
  dpi = 300
)

# -------------------------------------------------------------------------
# Pairwise win-count matrix
# -------------------------------------------------------------------------

wide <- plot_df %>%
  select(dataset, embedder, test_metric) %>%
  pivot_wider(
    names_from = embedder,
    values_from = test_metric
  )

models <- model_order

pairwise <- expand.grid(
  row_model = models,
  col_model = models,
  stringsAsFactors = FALSE
) %>%
  rowwise() %>%
  mutate(
    n = sum(!is.na(wide[[row_model]]) & !is.na(wide[[col_model]])),
    wins = sum(
      wide[[row_model]] > wide[[col_model]] + tie_threshold,
      na.rm = TRUE
    ),
    ties = sum(
      abs(wide[[row_model]] - wide[[col_model]]) <= tie_threshold,
      na.rm = TRUE
    ),
    losses = sum(
      wide[[row_model]] < wide[[col_model]] - tie_threshold,
      na.rm = TRUE
    ),
    win_score = wins + 0.5 * ties,
    win_rate = win_score / n,
    label = ifelse(
      row_model == col_model,
      "",
      paste0(win_score, "/", n)
    )
  ) %>%
  ungroup() %>%
  mutate(
    row_model = factor(row_model, levels = rev(models)),
    col_model = factor(col_model, levels = models)
  )

write_csv(
  pairwise,
  file.path(output_dir, "pairwise_win_counts.csv")
)

p_pairwise <- ggplot(
  pairwise,
  aes(x = col_model, y = row_model, fill = win_rate)
) +
  geom_tile(color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    size = 2.6
  ) +
  scale_fill_gradient2(
    low = "#b2182b",
    mid = "white",
    high = "#2166ac",
    midpoint = 0.5,
    limits = c(0, 1),
    name = "Win rate"
  ) +
  coord_fixed() +
  labs(
    x = NULL,
    y = NULL,
    title = "Pairwise dataset-level wins",
    subtitle = paste0(
      "Cell = row model wins over column model; ties count as 0.5. ",
      "Tie threshold = ",
      tie_threshold,
      "."
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(
  file.path(output_dir, "pairwise_win_count_matrix.pdf"),
  p_pairwise,
  width = 8,
  height = 7
)

ggsave(
  file.path(output_dir, "pairwise_win_count_matrix.png"),
  p_pairwise,
  width = 8,
  height = 7,
  dpi = 300
)

# -------------------------------------------------------------------------
# Optional: pairwise mean rank difference matrix
# -------------------------------------------------------------------------

rank_wide <- rank_df %>%
  mutate(embedder = as.character(embedder)) %>%
  select(dataset, embedder, rank) %>%
  pivot_wider(
    names_from = embedder,
    values_from = rank
  )

pairwise_rankdiff <- expand.grid(
  row_model = models,
  col_model = models,
  stringsAsFactors = FALSE
) %>%
  rowwise() %>%
  mutate(
    mean_rank_diff = mean(
      rank_wide[[row_model]] - rank_wide[[col_model]],
      na.rm = TRUE
    ),
    label = ifelse(
      row_model == col_model,
      "",
      sprintf("%.1f", mean_rank_diff)
    )
  ) %>%
  ungroup() %>%
  mutate(
    row_model = factor(row_model, levels = rev(models)),
    col_model = factor(col_model, levels = models)
  )

write_csv(
  pairwise_rankdiff,
  file.path(output_dir, "pairwise_rank_differences.csv")
)

p_rankdiff <- ggplot(
  pairwise_rankdiff,
  aes(x = col_model, y = row_model, fill = mean_rank_diff)
) +
  geom_tile(color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    size = 2.6
  ) +
  scale_fill_gradient2(
    low = "#2166ac",
    mid = "white",
    high = "#b2182b",
    midpoint = 0,
    name = "Mean rank\ndifference"
  ) +
  coord_fixed() +
  labs(
    x = NULL,
    y = NULL,
    title = "Pairwise mean rank differences",
    subtitle = "Negative values mean the row model ranks better than the column model"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(
  file.path(output_dir, "pairwise_rank_difference_matrix.pdf"),
  p_rankdiff,
  width = 8,
  height = 7
)

ggsave(
  file.path(output_dir, "pairwise_rank_difference_matrix.png"),
  p_rankdiff,
  width = 8,
  height = 7,
  dpi = 300
)

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

message("Wrote outputs to: ", output_dir)
message("Files:")
message("  - dataset_embedder_coverage.csv")
message("  - ranked_dataset_embedder_long.csv")
message("  - model_rank_summary.csv")
message("  - model_rank_dotplot.pdf/png")
message("  - pairwise_win_counts.csv")
message("  - pairwise_win_count_matrix.pdf/png")
message("  - pairwise_rank_differences.csv")
message("  - pairwise_rank_difference_matrix.pdf/png")


############################
library(dplyr)
library(ggplot2)
library(readr)

# rank_df should already contain:
# dataset, embedder, test_metric, rank

rank_summary <- rank_df %>%
  group_by(embedder) %>%
  summarise(
    mean_rank = mean(rank),
    median_rank = median(rank),
    q25 = quantile(rank, 0.25),
    q75 = quantile(rank, 0.75),
    .groups = "drop"
  ) %>%
  arrange(mean_rank) %>%
  mutate(embedder = factor(embedder, levels = rev(embedder)))

p <- ggplot(rank_summary, aes(x = mean_rank, y = embedder)) +
  geom_errorbarh(
    aes(xmin = q25, xmax = q75),
    height = 0,
    linewidth = 1.1,
    alpha = 0.7
  ) +
  geom_point(size = 3.2) +
  scale_x_reverse(
    breaks = seq(1, max(rank_summary$q75), by = 1)
  ) +
  labs(
    x = "Within-dataset rank (1 = best)",
    y = NULL,
    title = "Average model rank across benchmark datasets",
    subtitle = "Point = mean rank; horizontal bar = interquartile range"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave("model_mean_rank_iqr.pdf", p, width = 6.5, height = 4.5)
ggsave("model_mean_rank_iqr.png", p, width = 6.5, height = 4.5, dpi = 300)

set.seed(1)

boot_mean_rank <- function(x, n_boot = 5000) {
  replicate(n_boot, mean(sample(x, replace = TRUE)))
}

rank_summary_boot <- rank_df %>%
  group_by(embedder) %>%
  summarise(
    mean_rank = mean(rank),
    ci_low = quantile(boot_mean_rank(rank), 0.025),
    ci_high = quantile(boot_mean_rank(rank), 0.975),
    .groups = "drop"
  ) %>%
  arrange(mean_rank) %>%
  mutate(embedder = factor(embedder, levels = rev(embedder)))

p <- ggplot(rank_summary_boot, aes(x = mean_rank, y = embedder)) +
  geom_errorbarh(
    aes(xmin = ci_low, xmax = ci_high),
    height = 0,
    linewidth = 1.1,
    alpha = 0.75
  ) +
  geom_point(size = 3.2) +
  scale_x_reverse(
    breaks = seq(1, ceiling(max(rank_summary_boot$ci_high)), by = 1)
  ) +
  labs(
    x = "Mean within-dataset rank (1 = best)",
    y = NULL,
    title = "Mean model rank across benchmark datasets",
    subtitle = "Points show mean rank; bars show bootstrap 95% CI"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave("model_mean_rank_bootstrap_ci.pdf", p, width = 6.5, height = 4.5)
ggsave(
  "model_mean_rank_bootstrap_ci.png",
  p,
  width = 6.5,
  height = 4.5,
  dpi = 300
)


rank_heatmap_df <- rank_df %>%
  group_by(embedder) %>%
  mutate(mean_rank = mean(rank)) %>%
  ungroup() %>%
  group_by(dataset) %>%
  mutate(best_auc = max(test_metric, na.rm = TRUE)) %>%
  ungroup()

model_order <- rank_heatmap_df %>%
  distinct(embedder, mean_rank) %>%
  arrange(mean_rank) %>%
  pull(embedder)

dataset_order <- rank_heatmap_df %>%
  group_by(dataset) %>%
  summarise(
    mean_top_rank_auc = max(test_metric, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_top_rank_auc)) %>%
  pull(dataset)

rank_heatmap_df <- rank_heatmap_df %>%
  mutate(
    embedder = factor(embedder, levels = model_order),
    dataset = factor(dataset, levels = rev(dataset_order))
  )

p_heat <- ggplot(rank_heatmap_df, aes(x = embedder, y = dataset, fill = rank)) +
  geom_tile(color = "white", linewidth = 0.25) +
  scale_fill_viridis_c(
    option = "mako",
    direction = -1,
    breaks = seq(1, length(model_order), by = 1),
    name = "Rank"
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Within-dataset model ranks",
    subtitle = "Lower rank is better"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave("dataset_model_rank_heatmap.pdf", p_heat, width = 7.5, height = 6)
ggsave(
  "dataset_model_rank_heatmap.png",
  p_heat,
  width = 7.5,
  height = 6,
  dpi = 300
)

##################

#!/usr/bin/env Rscript

# dabestr_rank_plot.R
#
# DABEST-style paired estimation plot on within-dataset ranks.
#
# Input:
#   best_metric_by_dataset_embedder.csv
#
# Required columns:
#   dataset
#   embedder
#   test_metric
#
# Usage:
#   Rscript dabestr_rank_plot.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(dabestr)
})

# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------

input_csv <- "outputs/best_metric_by_dataset_embedder.csv"
output_dir <- "outputs/dabest"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Edit these to exact names in your CSV.
embedder_order <- c(
  "ECFP",
  "ECFP_count",
  "mol2vec",
  "MoLFormer-XL-both-10pct",
  "ChemBERTa-77M-MLM",
  "SELFormer",
  "CDDD",
  "CLAMP",
)

baseline_model <- "ECFP"

# DABEST comparison mode.
# "baseline" compares every model to the first model in idx.
# "sequential" compares model 2 - model 1, model 3 - model 2, etc.
paired_mode <- "baseline"

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

df <- read_csv(input_csv, show_col_types = FALSE)

required_cols <- c("dataset", "embedder", "test_metric")
missing_cols <- setdiff(required_cols, names(df))

if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- df %>%
  mutate(
    dataset = as.character(dataset),
    embedder = as.character(embedder),
    test_metric = as.numeric(test_metric)
  )

available <- sort(unique(df$embedder))
missing_embedders <- setdiff(embedder_order, available)

if (length(missing_embedders) > 0) {
  warning(
    "These embedders are not present in the CSV:\n  ",
    paste(missing_embedders, collapse = "\n  ")
  )
}

embedder_order <- intersect(embedder_order, available)

if (!(baseline_model %in% embedder_order)) {
  stop(
    "baseline_model is not present in embedder_order after filtering: ",
    baseline_model
  )
}

# Put baseline first, because DABEST baseline mode uses first idx element.
embedder_order <- c(
  baseline_model,
  setdiff(embedder_order, baseline_model)
)

# -------------------------------------------------------------------------
# Compute within-dataset ranks
# -------------------------------------------------------------------------

rank_df <- df %>%
  filter(embedder %in% embedder_order) %>%
  select(dataset, embedder, test_metric) %>%
  filter(!is.na(test_metric)) %>%
  group_by(dataset) %>%
  filter(n_distinct(embedder) == length(embedder_order)) %>%
  mutate(
    rank = rank(-test_metric, ties.method = "average")
  ) %>%
  ungroup() %>%
  mutate(
    embedder = factor(embedder, levels = embedder_order),
    dataset = factor(dataset)
  )

if (nrow(rank_df) == 0) {
  stop("No complete datasets left after filtering. Check selected embedders.")
}

message("Using ", n_distinct(rank_df$dataset), " complete datasets.")
message("Using ", n_distinct(rank_df$embedder), " embedders.")

write_csv(
  rank_df,
  file.path(output_dir, "dabestr_rank_input_long.csv")
)

# -------------------------------------------------------------------------
# Sanity summaries
# -------------------------------------------------------------------------

rank_summary <- rank_df %>%
  group_by(embedder) %>%
  summarise(
    n = n(),
    mean_rank = mean(rank),
    median_rank = median(rank),
    mean_auc = mean(test_metric),
    .groups = "drop"
  ) %>%
  arrange(mean_rank)

write_csv(
  rank_summary,
  file.path(output_dir, "rank_summary.csv")
)

print(rank_summary)

# Paired mean rank differences vs baseline, independent of DABEST.
rank_wide <- rank_df %>%
  select(dataset, embedder, rank) %>%
  pivot_wider(names_from = embedder, values_from = rank)

rank_diffs <- rank_wide %>%
  summarise(
    across(
      all_of(setdiff(embedder_order, baseline_model)),
      ~ mean(.x - .data[[baseline_model]], na.rm = TRUE)
    )
  ) %>%
  pivot_longer(
    everything(),
    names_to = "embedder",
    values_to = "mean_rank_diff_vs_baseline"
  ) %>%
  arrange(mean_rank_diff_vs_baseline)

write_csv(
  rank_diffs,
  file.path(output_dir, "paired_rank_differences_vs_baseline.csv")
)

print(rank_diffs)

# -------------------------------------------------------------------------
# DABEST paired rank analysis
# -------------------------------------------------------------------------

dabest_rank <- load(
  data = rank_df,
  x = embedder,
  y = rank,
  idx = embedder_order,
  paired = paired_mode,
  id_col = dataset
) %>%
  mean_diff()

capture.output(
  print(dabest_rank),
  file = file.path(output_dir, "dabestr_rank_mean_diff_summary.txt")
)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

p <- dabest_plot(
  dabest_rank,
  raw_marker_size = 1.4,
  raw_marker_alpha = 0.45,
  float_contrast = FALSE
)

# DABEST returns a ggplot-like object. This usually works.
# Because rank 1 is best, reverse the raw-data rank axis if possible.
p <- p +
  labs(
    title = "Paired DABEST plot of within-dataset ranks",
    subtitle = paste0(
      "Ranks computed within each dataset. Baseline: ",
      baseline_model,
      ". Negative differences mean better rank than baseline."
    ),
    y = "Within-dataset rank"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 35, hjust = 1)
  )

ggsave(
  file.path(output_dir, "dabestr_rank_mean_diff.pdf"),
  p,
  width = 9.5,
  height = 6
)

ggsave(
  file.path(output_dir, "dabestr_rank_mean_diff.png"),
  p,
  width = 9.5,
  height = 6,
  dpi = 300
)

message("Wrote outputs to: ", output_dir)

rank_df <- rank_df %>%
  mutate(
    model_group = ifelse(
      grepl("ModernMolBERT", embedder),
      "ModernMolBERT",
      "Comparator"
    )
  )

ggplot(rank_df, aes(x = dataset, y = rank, group = embedder)) +
  geom_line(
    data = subset(rank_df, model_group == "Comparator"),
    aes(color = embedder),
    linewidth = 0.35,
    alpha = 0.35
  ) +
  geom_line(
    data = subset(rank_df, model_group == "ModernMolBERT"),
    aes(color = embedder),
    linewidth = 0.9,
    alpha = 0.95
  ) +
  geom_point(
    data = subset(rank_df, model_group == "ModernMolBERT"),
    aes(color = embedder),
    size = 1.6
  ) +
  scale_y_reverse(
    breaks = seq(1, length(unique(rank_df$embedder)), by = 1)
  ) +
  labs(
    x = "Dataset",
    y = "Within-dataset rank (1 = best)",
    title = "Model rank trajectories across datasets"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank()
  )
