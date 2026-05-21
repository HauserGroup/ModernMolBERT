library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

data_path <- "runs/chembl36_small_mask_mlm_lr_sweep/sweep_results.csv"
out_dir   <- "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

raw <- read_csv(data_path, show_col_types = FALSE)

df <- raw |>
  filter(strategy %in% c("standard", "span")) |>
  mutate(
    strategy = factor(strategy, levels = c("standard", "span")),
    lr_label = case_when(
      learning_rate == 1e-4 ~ "1e-4",
      learning_rate == 2e-4 ~ "2e-4",
      learning_rate == 4e-4 ~ "4e-4"
    ),
    lr_label = factor(lr_label, levels = c("1e-4", "2e-4", "4e-4"))
  ) |>
  pivot_longer(
    cols      = c(eval_loss, eval_masked_accuracy),
    names_to  = "metric",
    values_to = "value"
  ) |>
  mutate(
    metric = factor(
      metric,
      levels = c("eval_masked_accuracy", "eval_loss"),
      labels = c("Masked accuracy", "Eval loss")
    )
  )

strategy_colors   <- c(standard = "#2171B5", span = "#D94801")
lr_linetypes      <- c("1e-4" = "dotted", "2e-4" = "dashed", "4e-4" = "solid")
lr_shapes         <- c("1e-4" = 1, "2e-4" = 17, "4e-4" = 16)

p <- ggplot(
  df,
  aes(
    x      = mlm_prob,
    y      = value,
    color  = strategy,
    linetype = lr_label,
    shape  = lr_label,
    group  = interaction(strategy, lr_label)
  )
) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.5) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_x_continuous(
    breaks = sort(unique(df$mlm_prob)),
    labels = scales::label_percent(accuracy = 1)
  ) +
  scale_color_manual(
    values = strategy_colors,
    labels = c(standard = "Standard", span = "Span"),
    name   = "Strategy"
  ) +
  scale_linetype_manual(values = lr_linetypes, name = "Learning rate") +
  scale_shape_manual(values  = lr_shapes,     name = "Learning rate") +
  labs(
    x        = "Trained MLM probability",
    y        = NULL,
    title    = "MLM sweep: all masking probabilities × learning rates",
    subtitle = "Evaluated on each model’s own validation set"
  ) +
  theme_classic(base_size = 11) +
  theme(
    strip.background   = element_blank(),
    strip.text         = element_text(face = "bold", size = 10),
    legend.position    = "bottom",
    legend.box         = "horizontal",
    legend.margin      = margin(t = 4),
    panel.spacing.x    = unit(1.6, "lines"),
    plot.title         = element_text(face = "bold"),
    plot.subtitle      = element_text(size = 9, color = "grey40")
  )

ggsave(file.path(out_dir, "FigX_sweep_all.pdf"), p, width = 9, height = 4.2)
ggsave(file.path(out_dir, "FigX_sweep_all.png"), p, width = 9, height = 4.2, dpi = 300)
message("Wrote figures/FigX_sweep_all.{pdf,png}")
