import sys
from pathlib import Path

import pytest


def test_list_featurizers_cli_prints_registered_featurizers(capsys) -> None:
    from modernmolbert.eval.cli import list_featurizers

    list_featurizers.main()

    out = capsys.readouterr().out
    assert "dummy\t[core]" in out
    assert "ecfp4\t[eval-rdkit]" in out


def test_prepare_moleculenet_cli_lists_datasets_without_preparing(monkeypatch, capsys) -> None:
    from modernmolbert.eval.cli import prepare_moleculenet

    def fail_prepare_many(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("--list_datasets should return before prepare_many")

    monkeypatch.setattr(prepare_moleculenet, "prepare_many", fail_prepare_many)
    monkeypatch.setattr(sys, "argv", ["prepare_moleculenet.py", "--list_datasets"])

    prepare_moleculenet.main()

    out = capsys.readouterr().out
    assert "Core datasets:" in out
    assert "Extended datasets:" in out


def test_prepare_moleculenet_cli_rejects_keep_invalid_with_scaffold(monkeypatch) -> None:
    from modernmolbert.eval.cli import prepare_moleculenet

    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_moleculenet.py", "--datasets", "bbbp", "--keep_invalid", "--split", "scaffold"],
    )

    with pytest.raises(SystemExit, match="--keep_invalid cannot be used with --split scaffold"):
        prepare_moleculenet.main()


def test_report_benchmark_results_no_plots_uses_table_writer_only(
    monkeypatch, tmp_path, capsys
) -> None:
    from modernmolbert.eval.cli import report_benchmark_results

    calls = []

    def fake_write_summary_tables(*, results_path: Path, output_dir: Path):
        calls.append(("tables", results_path, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        return [output_dir / "summary.csv"]

    def fake_write_standard_plots(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("--no_plots should skip plot generation")

    results_csv = tmp_path / "results.csv"
    results_csv.write_text("dataset,task\n", encoding="utf-8")
    output_dir = tmp_path / "report"
    monkeypatch.setattr(report_benchmark_results, "write_summary_tables", fake_write_summary_tables)
    monkeypatch.setattr(report_benchmark_results, "write_standard_plots", fake_write_standard_plots)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_benchmark_results.py",
            "--results_csv",
            str(results_csv),
            "--output_dir",
            str(output_dir),
            "--no_plots",
        ],
    )

    report_benchmark_results.main()

    assert calls == [("tables", results_csv, output_dir / "tables")]
    out = capsys.readouterr().out
    assert "Wrote 1 summary table" in out
    assert "Done." in out
