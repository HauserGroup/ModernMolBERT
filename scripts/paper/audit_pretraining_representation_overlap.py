"""Compare benchmark test SELFIES with ChEMBL training SELFIES for matched molecules.

This is an input/provenance audit, not a model evaluation. An InChIKey can group
different SMILES (including tautomers), so a mismatch under that key does not
by itself identify a sanitisation bug. Exact canonical-SMILES matches are
reported separately.

Usage:
    uv run python scripts/paper/audit_pretraining_representation_overlap.py \
        --summary-output outputs/audit/pretraining_representation_summary.csv \
        --row-output outputs/audit/pretraining_representation_test_rows.csv
"""

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
import selfies as sf
import yaml

from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.utils import file_sha256 as sha256_file

CONFIG = Path("src/modernmolbert/eval/benchmarking_molecular_models/config/datasets.yaml")
PRETRAIN = Path("data/pretrain/chembl36_selfies/train.parquet")
PREPARED = Path("data/prepared")
EXCLUDED = {"ogbg-moltoxcast"}  # The 26th configured dataset is not in the paper.


@lru_cache(maxsize=250_000)
def inspect_smiles(smiles: str) -> tuple[str | None, str | None]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        key = Chem.MolToInchiKey(mol) or None
    except Exception:
        return None, None
    try:
        encoded = sf.encoder(smiles) or None
    except Exception:
        encoded = None
    return key, encoded


def test_rows(config: Path, prepared_dir: Path) -> list[dict[str, object]]:
    entries = yaml.safe_load(config.read_text())["datasets"]
    rows: list[dict[str, object]] = []
    for entry in entries.values():
        name = entry["name"]
        if name in EXCLUDED:
            continue
        path = prepared_dir / f"{name}.json"
        dataset = Dataset.deserialize_legacy(path)
        prepared_hash = sha256_file(path)
        for index in dataset.splits["test"]:
            position = int(index)
            raw = dataset.data.iloc[position]["smiles"]
            smiles = raw if isinstance(raw, str) else ""
            key, encoded = inspect_smiles(smiles)
            rows.append(
                {
                    "dataset": name,
                    "test_row_index": position,
                    "prepared_sha256": prepared_hash,
                    "smiles": smiles,
                    "standard_inchi_key": key or "",
                    "eval_selfies": encoded or "",
                }
            )
    return rows


def pretrain_matches(
    path: Path, target_keys: set[str], target_smiles: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    by_key_selfies: dict[str, set[str]] = defaultdict(set)
    by_key_smiles: dict[str, set[str]] = defaultdict(set)
    by_smiles_selfies: dict[str, set[str]] = defaultdict(set)
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(
        batch_size=100_000,
        columns=["standard_inchi_key", "smiles_canonical_clean", "selfies"],
    ):
        block = batch.to_pydict()
        for key, smiles, encoded in zip(
            block["standard_inchi_key"],
            block["smiles_canonical_clean"],
            block["selfies"],
            strict=True,
        ):
            if not isinstance(encoded, str):
                continue
            if isinstance(key, str) and key in target_keys:
                by_key_selfies[key].add(encoded)
                if isinstance(smiles, str):
                    by_key_smiles[key].add(smiles)
            if isinstance(smiles, str) and smiles in target_smiles:
                by_smiles_selfies[smiles].add(encoded)
    return by_key_selfies, by_key_smiles, by_smiles_selfies


def annotate_rows(
    rows: list[dict[str, object]],
    by_key_selfies: dict[str, set[str]],
    by_key_smiles: dict[str, set[str]],
    by_smiles_selfies: dict[str, set[str]],
) -> None:
    for row in rows:
        key = str(row["standard_inchi_key"])
        smiles = str(row["smiles"])
        encoded = str(row["eval_selfies"])
        key_values = by_key_selfies.get(key, set()) if key else set()
        smiles_values = by_smiles_selfies.get(smiles, set()) if smiles else set()
        row.update(
            {
                "matched_pretrain_inchikey": bool(key_values),
                "matched_pretrain_canonical_smiles": bool(smiles_values),
                "selfies_match_at_inchikey": bool(encoded and encoded in key_values),
                "selfies_match_at_canonical_smiles": bool(encoded and encoded in smiles_values),
                "n_pretrain_smiles_at_inchikey": len(by_key_smiles.get(key, set())),
                "n_pretrain_selfies_at_inchikey": len(key_values),
            }
        )


def summaries(rows: list[dict[str, object]], pretrain_sha256: str) -> list[dict[str, object]]:
    names = sorted({str(row["dataset"]) for row in rows})
    groups = [(name, [row for row in rows if row["dataset"] == name]) for name in names]
    groups.append(("ALL_PAPER_TEST_ROWS", rows))
    result = []
    for name, group in groups:
        result.append(
            {
                "dataset": name,
                "pretrain_sha256": pretrain_sha256,
                "n_test_rows": len(group),
                "n_eval_selfies_failed": sum(not row["eval_selfies"] for row in group),
                "n_inchikey_overlap": sum(bool(row["matched_pretrain_inchikey"]) for row in group),
                "n_canonical_smiles_overlap": sum(
                    bool(row["matched_pretrain_canonical_smiles"]) for row in group
                ),
                "n_identifier_overlap": sum(
                    bool(row["matched_pretrain_inchikey"])
                    or bool(row["matched_pretrain_canonical_smiles"])
                    for row in group
                ),
                "n_inchikey_only_overlap": sum(
                    bool(row["matched_pretrain_inchikey"])
                    and not row["matched_pretrain_canonical_smiles"]
                    for row in group
                ),
                "n_canonical_smiles_only_overlap": sum(
                    bool(row["matched_pretrain_canonical_smiles"])
                    and not row["matched_pretrain_inchikey"]
                    for row in group
                ),
                "n_inchikey_overlap_selfies_match": sum(
                    bool(row["selfies_match_at_inchikey"]) for row in group
                ),
                "n_inchikey_overlap_selfies_differ": sum(
                    bool(row["matched_pretrain_inchikey"])
                    and bool(row["eval_selfies"])
                    and not row["selfies_match_at_inchikey"]
                    for row in group
                ),
                "n_canonical_smiles_overlap_selfies_match": sum(
                    bool(row["selfies_match_at_canonical_smiles"]) for row in group
                ),
                "n_canonical_smiles_overlap_selfies_differ": sum(
                    bool(row["matched_pretrain_canonical_smiles"])
                    and bool(row["eval_selfies"])
                    and not row["selfies_match_at_canonical_smiles"]
                    for row in group
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED)
    parser.add_argument("--pretrain", type=Path, default=PRETRAIN)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--row-output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    rows = test_rows(args.config, args.prepared_dir)
    targets_keys = {str(row["standard_inchi_key"]) for row in rows if row["standard_inchi_key"]}
    targets_smiles = {str(row["smiles"]) for row in rows if row["smiles"]}
    by_key, by_key_smiles, by_smiles = pretrain_matches(args.pretrain, targets_keys, targets_smiles)
    annotate_rows(rows, by_key, by_key_smiles, by_smiles)
    summary = summaries(rows, sha256_file(args.pretrain))
    write_csv(args.row_output, rows)
    write_csv(args.summary_output, summary)
    print(summary[-1])


if __name__ == "__main__":
    main()
