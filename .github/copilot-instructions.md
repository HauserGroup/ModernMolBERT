---
applyTo: "*.R/*.qmd/*.rmd"
---

# Terminal use
Never use terminal or zsh commands to run or inspect R or Quarto files. Always just edit the notebook directly,

# Preferred function use in R
All library calls should be at the top of the notebook, and should be tidyverse packages when possible. Avoid loading base R packages like `stats` or `graphics` unless absolutely necessary.
Use tidyverse functions and style when possible. Avoid base R functions and syntax, especially for data manipulation and visualization.
Use |> for piping instead of %>% when possible.
Use scale_x_continuous and scale_y_continuous for setting axis limits and breaks in ggplot2, instead of xlim(), ylim(), or coord_cartesian().
Always use return() to explicitly return values from functions, even if it's not strictly necessary. This improves readability and makes it clear what the output of the function is.
Always reconsider whether && or & should be use inside mutate calls

# Preferred plotting style in R
Use `ggplot2` for plotting instead of base R plotting functions.
Use viridisLite for continuous scale data color and fill.
Use RColorBrewer color palettes to scale discrete data color and fill rather than defining manually.
Use theme_classic().
Limit label and title length and be concise.
Do not add captions to plots unless necessary for clarity.
Use labs() to set plot titles, axis labels, and legends in a single call.

# Editing style in R
Do not remove Quarto code chunks i.e. ```{r} ... ``` or ```{python} ... ```. Always keep the code chunks intact and only edit the code within them.

# Common errors in R
Do not use && or || for logical operations inside mutate calls. Always use & instead, as && only evaluates the first element of each vector and can lead to unexpected results when used with dplyr functions.

Do not use isTrue() or isFALSE() inside mutate calls. Instead, use direct logical comparisons (e.g., variable == TRUE or variable == FALSE) to ensure proper evaluation of logical conditions within dplyr pipelines.

Purrr does have .progress implemented and it is fine to use

Trailing commas in function calls is fine and can improve readability, so do not remove them. They are not fine in vector definitions, so do remove them from c() calls.

# Comments
Don't remove comments from the code, but feel free to edit them for clarity and conciseness. Always keep comments that explain the purpose of code blocks or complex logic, as they can be helpful for future reference and for other readers of the code.

# Python
"from __future__ import annotations" is not necessary in Python 3.10 and later, so do not add it to any new Python files. It is only needed for backward compatibility with older Python versions.
Always implement notebooks as py:percent and not Jupyter notebooks, .ipynb files. This means using the `# %%` syntax to define code cells, and avoiding the use of Jupyter-specific features like markdown cells or magic commands. This allows for better compatibility with different editors and tools, and makes it easier to version control the code.
