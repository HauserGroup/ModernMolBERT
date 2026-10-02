"""The paper matrix must preserve matched internal and descriptive external cohorts."""

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_revision_paper_results_use_common_internal_rows(tmp_path: Path) -> None:
    tasks = ["Bioavailability_Ma", "AMES"]
    baselines = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"]
    internal = [
        "MMB-small-APE-SELFIES",
        "MMB-small-APE-SMILES",
        "MMB-small-BPE-SELFIES",
        "MMB-small-BPE-SMILES",
        "MMB-base-APE-SELFIES",
    ]
    native = pd.DataFrame(0.6, index=tasks, columns=baselines + internal)
    native.loc[:, baselines] = 0.8
    common = pd.DataFrame(0.7, index=tasks, columns=internal)
    native_path = tmp_path / "task_matrix.csv"
    common_path = tmp_path / "common_task_matrix.csv"
    out = tmp_path / "paper"
    native.to_csv(native_path)
    common.to_csv(common_path)
    aggregate_manifest = tmp_path / "manifest.json"
    aggregate_manifest.write_text(
        json.dumps(
            {
                "seeds": [42, 43, 44, 45, 46],
                "models": internal,
                "task_count": 25,
                "output_sha256": {
                    "mean_common_task_matrix.csv": hashlib.sha256(
                        common_path.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    mean_path = tmp_path / "mean_common_task_matrix.csv"
    common_path.rename(mean_path)
    common_path = mean_path

    script = Path(__file__).resolve().parents[1] / "scripts/paper/build_paper_results.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--task-matrix",
            str(native_path),
            "--common-task-matrix",
            str(common_path),
            "--seed-aggregate-manifest",
            str(aggregate_manifest),
            "--reference",
            internal[0],
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    matrix = pd.read_csv(out / "results_matrix_25task.csv", index_col=0)
    assert matrix.loc["AMES", internal[0]] == 0.7
    assert matrix.loc["AMES", "ECFP4"] == 0.8
    provenance = json.loads((out / "results_matrix_provenance.json").read_text())
    assert provenance["internal_models"]["population"] == (
        "five-seed mean of five-model common test-row scores"
    )
    assert "means across five training seeds" in (out / "table2.tex").read_text()
    assert "unmatched test molecules" in result.stdout
    assert "Wilcoxon W=" not in result.stdout
    assert "MMB-small-BPE-SELFIES" in result.stdout

    figures = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(script.with_name("make_paper_figures.py")),
            "--matrix",
            str(out / "results_matrix_25task.csv"),
            "--reference",
            internal[0],
            "--figure-dir",
            str(figures),
            "--source-data-dir",
            str(out / "source_data"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (figures / "Fig_2.pdf").is_file()
    assert (figures / "Fig_groupbars.pdf").is_file()

    common_path.write_text(common_path.read_text() + "\n", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--task-matrix",
            str(native_path),
            "--common-task-matrix",
            str(common_path),
            "--seed-aggregate-manifest",
            str(aggregate_manifest),
            "--reference",
            internal[0],
            "--out-dir",
            str(tmp_path / "rejected"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "does not match the verified five-seed aggregate" in rejected.stderr
