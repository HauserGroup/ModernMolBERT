library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

sweep_path <- "runs/chembl36_small_mask_mlm_lr_sweep/sweep_results.csv"
out_dir <- "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

metric_labels <- c(
  eval_loss = "Evaluation loss",
  eval_masked_accuracy = "Masked-token accuracy",
  eval_perplexity = "Perplexity"
)

sweep <- read_csv(sweep_path, show_col_types = FALSE) |>
  filter(strategy != "hetero_span") |>
  mutate(
    mlm_probability = mlm_prob,
    learning_rate_num = learning_rate,
    learning_rate = factor(
      learning_rate,
      levels = sort(unique(learning_rate)),
      labels = scales::label_scientific(digits = 1)(sort(unique(
        learning_rate
      )))
    ),
    strategy = factor(strategy, levels = c("standard", "span")),
    curve = interaction(
      strategy,
      learning_rate,
      sep = " / ",
      lex.order = TRUE
    )
  ) |>
  pivot_longer(
    cols = c(eval_loss, eval_masked_accuracy, eval_perplexity),
    names_to = "metric",
    values_to = "value"
  ) |>
  mutate(
    metric = factor(
      metric,
      levels = names(metric_labels),
      labels = metric_labels
    )
  )

curve_levels <- sweep |>
  distinct(strategy, learning_rate_num, curve) |>
  arrange(strategy, learning_rate_num) |>
  pull(curve)

standard_levels <- curve_levels[str_starts(
  as.character(curve_levels),
  "standard"
)]
span_levels <- curve_levels[str_starts(as.character(curve_levels), "span")]

curve_colors <- c(
  setNames(
    colorRampPalette(RColorBrewer::brewer.pal(5, "Blues")[3:5])(
      length(standard_levels)
    ),
    standard_levels
  ),
  setNames(
    colorRampPalette(RColorBrewer::brewer.pal(5, "Oranges")[3:5])(
      length(span_levels)
    ),
    span_levels
  )
)

p <- ggplot(
  sweep,
  aes(
    x = mlm_probability,
    y = value,
    color = curve,
    group = curve
  )
) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.4) +
  facet_grid(curve ~ metric, scales = "free_y") +
  scale_x_continuous(
    breaks = sort(unique(sweep$mlm_probability)),
    labels = scales::label_percent(accuracy = 1)
  ) +
  scale_color_manual(
    values = curve_colors,
    breaks = curve_levels,
    guide = "none"
  ) +
  labs(
    x = "MLM probability",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.spacing.x = unit(1.4, "lines")
  )

ggsave(
  file.path(out_dir, "FigX_sweep_metrics.pdf"),
  p,
  width = 8.2,
  height = 8.8
)
ggsave(
  file.path(out_dir, "FigX_sweep_metrics.png"),
  p,
  width = 8.2,
  height = 8.8,
  dpi = 300
)
