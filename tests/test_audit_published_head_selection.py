import pandas as pd

from audit_published_head_selection import main

TABLE = r"""
\begin{longtable}{l r r }
  \textbf{Task} & \textbf{ECFP4} & \textbf{MMB-s} \\
  \endhead
  \multicolumn{3}{l}{\textit{TDC -- Toxicity}} \\
  AMES & 90.0 & 80.0 \\
  DILI & 70.0 & -- \\
\end{longtable}
"""


def _row(dataset, embedder, model, cv, test):
    return {
        "dataset": dataset,
        "embedder": embedder,
        "model": model,
        "cv_metric_name": "roc_auc",
        "cv_metric": cv,
        "test_metric_name": "roc_auc",
        "test_metric": test,
    }


def test_published_cells_are_classified_by_selection_rule(tmp_path):
    table = tmp_path / "pertask.tex"
    table.write_text(TABLE)
    praski = tmp_path / "praski.csv"
    pd.DataFrame(
        [
            # Test maximum (0.90) is not the CV winner (0.85).
            _row("AMES", "ECFP", "rf", 0.80, 0.90),
            _row("AMES", "ECFP", "knn", 0.90, 0.85),
            # One head: test maximum and CV winner coincide.
            _row("DILI", "ECFP", "rf", 0.80, 0.70),
        ]
    ).to_csv(praski, index=False)
    own = tmp_path / "eval" / "praski_best_standard" / "results.csv"
    own.parent.mkdir(parents=True)
    # The recorded embedder label is overridden by the directory, as historically.
    pd.DataFrame(
        [
            _row("AMES", "anything", "ridge", 0.70, 0.75),
            _row("AMES", "anything", "rf", 0.60, 0.80),
        ]
    ).to_csv(own, index=False)

    out = tmp_path / "audit.csv"
    main(
        [
            "--table",
            str(table),
            "--praski-csv",
            str(praski),
            "--results-root",
            str(tmp_path / "eval"),
            "--output",
            str(out),
        ]
    )
    audit = pd.read_csv(out).set_index(["dataset", "table_column"])
    assert audit.loc[("AMES", "ECFP4"), "rule"] == "test_max_only"
    assert audit.loc[("AMES", "ECFP4"), "cv_selected_head"] == "knn"
    assert audit.loc[("DILI", "ECFP4"), "rule"] == "both"
    assert audit.loc[("AMES", "MMB-s"), "rule"] == "test_max_only"
    assert audit.loc[("AMES", "MMB-s"), "embedder"] == "modernmolbert_best_standard"
    assert audit.loc[("DILI", "MMB-s"), "rule"] == "missing"
