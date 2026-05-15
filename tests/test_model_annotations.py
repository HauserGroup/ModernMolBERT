import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.model_annotations import (
    annotate_model_name,
    annotate_model_table,
    strip_variant_suffix,
    summarize_model_categories,
)


def test_strip_variant_suffix_removes_bracketed_parts() -> None:
    assert strip_variant_suffix("ChemBERTa_[10M][MTR]") == "ChemBERTa"
    assert strip_variant_suffix("ECFP_[Count]") == "ECFP"
    assert strip_variant_suffix("MoLFormer_[100M]") == "MoLFormer"


def test_annotate_ecfp_as_fingerprint() -> None:
    ann = annotate_model_name("ECFP_[Count]")

    assert ann["representation_family"] == "fingerprint"
    assert ann["baseline_role"] == "classical baseline"
    assert ann["comparable_to_ours"] == "essential baseline, not architecture-comparable"


def test_annotate_chemberta_as_string_lm() -> None:
    ann = annotate_model_name("ChemBERTa_[77M][MLM]")

    assert ann["representation_family"] == "string_lm"
    assert ann["input_modality"] == "SMILES"
    assert ann["comparable_to_ours"] == "yes"


def test_annotate_selfies_as_selfies_string_lm() -> None:
    ann = annotate_model_name("SELFormer")

    assert ann["representation_family"] == "string_lm"
    assert ann["input_modality"] == "SELFIES"
    assert ann["baseline_role"] == "closest representation comparable"


def test_annotate_modernmolbert_as_string_lm() -> None:
    ann = annotate_model_name("ModernMolBERT_SELFIES_ChEMBL36_2M")

    assert ann["representation_family"] == "string_lm"
    assert ann["model_class"] == "molecular language model"
    assert ann["comparable_to_ours"] == "yes"


def test_unknown_model_gets_unknown_category() -> None:
    ann = annotate_model_name("SomeNewUnseenModel")

    assert ann["representation_family"] == "unknown"
    assert ann["baseline_role"] == "needs manual annotation"


def test_annotate_model_table_uses_embedder_column() -> None:
    df = pd.DataFrame(
        {
            "dataset": ["AMES", "AMES"],
            "embedder": ["ECFP", "ChemBERTa"],
            "test_metric": [0.8, 0.82],
        }
    )

    out = annotate_model_table(df)

    assert "representation_family" in out.columns
    assert list(out["representation_family"]) == ["fingerprint", "string_lm"]


def test_annotate_model_table_uses_model_column() -> None:
    df = pd.DataFrame(
        {
            "Model": ["ECFP", "ModernMolBERT_SELFIES_ChEMBL36_2M"],
            "rank_best": [2.0, 1.0],
        }
    )

    out = annotate_model_table(df, model_col="Model")

    assert "model_base" in out.columns
    assert "representation_family" in out.columns
    assert list(out["representation_family"]) == ["fingerprint", "string_lm"]


def test_summarize_model_categories_deduplicates_models() -> None:
    df = pd.DataFrame(
        {
            "embedder": ["ECFP", "ECFP", "ChemBERTa", "ChemBERTa"],
            "dataset": ["A", "B", "A", "B"],
            "test_metric": [0.8, 0.7, 0.82, 0.72],
        }
    )

    out = summarize_model_categories(df)

    assert len(out) == 2
    assert set(out["model_base"]) == {"ECFP", "ChemBERTa"}
    assert set(out["representation_family"]) == {"fingerprint", "string_lm"}
