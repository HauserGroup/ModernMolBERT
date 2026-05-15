# pyright: reportArgumentType=false

import re
from pathlib import Path

import pandas as pd


MODEL_ANNOTATIONS = [
    # Classical fingerprints / descriptors
    {
        "pattern": r"^ECFP|Morgan|FCFP|AtomPair|TopologicalTorsion|MACCS|RDKit|Avalon|Pattern",
        "representation_family": "fingerprint",
        "input_modality": "2D structure",
        "model_class": "handcrafted fingerprint",
        "baseline_role": "classical baseline",
        "comparable_to_ours": "essential baseline, not architecture-comparable",
        "notes": "Strong non-neural molecular fingerprint baseline.",
    },
    {
        "pattern": r"Mordred|Descriptor|PhysChem|RDKitDescriptors",
        "representation_family": "descriptor",
        "input_modality": "2D structure",
        "model_class": "handcrafted descriptor",
        "baseline_role": "classical baseline",
        "comparable_to_ours": "baseline, not architecture-comparable",
        "notes": "Computed molecular descriptors rather than learned sequence embeddings.",
    },
    # SMILES / SELFIES language models
    {
        "pattern": r"ChemBERTa|ChemBERTa-2|SMILES.?BERT|MolBERT|MolRoBERTa|MegaMolBART|MolGPT|MolT5|MoleculeSTM",
        "representation_family": "string_lm",
        "input_modality": "SMILES",
        "model_class": "molecular language model",
        "baseline_role": "direct neural comparable",
        "comparable_to_ours": "yes",
        "notes": "SMILES-based pretrained molecular language model.",
    },
    {
        "pattern": r"MoLFormer|MolFormer",
        "representation_family": "string_lm",
        "input_modality": "SMILES",
        "model_class": "molecular language model",
        "baseline_role": "direct neural comparable",
        "comparable_to_ours": "yes",
        "notes": "Large-scale SMILES transformer baseline.",
    },
    {
        "pattern": r"SELFormer|SELFIES",
        "representation_family": "string_lm",
        "input_modality": "SELFIES",
        "model_class": "molecular language model",
        "baseline_role": "closest representation comparable",
        "comparable_to_ours": "yes",
        "notes": "SELFIES-based pretrained molecular language model.",
    },
    {
        "pattern": r"CDDD",
        "representation_family": "string_autoencoder",
        "input_modality": "SMILES/InChI",
        "model_class": "sequence autoencoder",
        "baseline_role": "direct neural comparable",
        "comparable_to_ours": "yes",
        "notes": "Continuous molecular descriptor learned from string reconstruction.",
    },
    {
        "pattern": r"Mol2Vec|mol2vec",
        "representation_family": "substructure_embedding",
        "input_modality": "2D structure / substructure tokens",
        "model_class": "word2vec-style molecular embedding",
        "baseline_role": "older learned baseline",
        "comparable_to_ours": "partly",
        "notes": "Learns embeddings of molecular substructures rather than full transformer sequence embeddings.",
    },
    # Graph models
    {
        "pattern": r"Grover|GROVER|GraphMVP|MolCLR|GraphCL|DimeNet|GIN|GNN|MPNN|GEM|MolGNet|Graphormer",
        "representation_family": "graph",
        "input_modality": "2D molecular graph",
        "model_class": "graph neural network",
        "baseline_role": "neural non-string comparable",
        "comparable_to_ours": "partly",
        "notes": "Graph-based learned representation; useful comparison but not same input representation.",
    },
    # 3D / geometry models
    {
        "pattern": r"Uni.?Mol|SchNet|PaiNN|TorchMD|DimeNet|GemNet|SphereNet|3D|Conformer",
        "representation_family": "geometry",
        "input_modality": "3D geometry/conformers",
        "model_class": "3D molecular representation model",
        "baseline_role": "neural non-string comparable",
        "comparable_to_ours": "partly",
        "notes": "Uses 3D molecular geometry or conformers; not a pure SMILES/SELFIES baseline.",
    },
    # Multimodal / bioactivity / protein-ligand
    {
        "pattern": r"CLAMP",
        "representation_family": "fingerprint_multimodal",
        "input_modality": "fingerprint + assay/text",
        "model_class": "contrastive multimodal model",
        "baseline_role": "strong but not direct architecture comparable",
        "comparable_to_ours": "partly",
        "notes": "Strong benchmark model; based on fingerprints and contrastive assay/text supervision.",
    },
    {
        "pattern": r"KV.?PLM|MolFM|BioT5|MoleculeSTM|Text2Mol|Protein|DTI",
        "representation_family": "multimodal",
        "input_modality": "molecule + text/protein/knowledge",
        "model_class": "multimodal pretrained model",
        "baseline_role": "contextual neural comparable",
        "comparable_to_ours": "partly",
        "notes": "Uses additional non-molecular-string supervision or modalities.",
    },
    # Your model
    {
        "pattern": r"ModernMolBERT|ModernBERT|ModernMol|MolEncoder",
        "representation_family": "string_lm",
        "input_modality": "SELFIES/SMILES",
        "model_class": "molecular language model",
        "baseline_role": "ours or closest family",
        "comparable_to_ours": "yes",
        "notes": "BERT-style masked molecular language model.",
    },
]


ANNOTATION_COLUMNS = [
    "representation_family",
    "input_modality",
    "model_class",
    "baseline_role",
    "comparable_to_ours",
    "notes",
]


def strip_variant_suffix(name: str) -> str:
    """Remove bracketed training/objective suffixes while keeping the raw name elsewhere."""
    name = str(name)
    name = re.sub(r"_?\[[^\]]+\]", "", name)
    return name


def annotate_model_name(name: str) -> dict[str, str]:
    raw = str(name)
    compact = strip_variant_suffix(raw)

    for entry in MODEL_ANNOTATIONS:
        if re.search(entry["pattern"], raw, flags=re.IGNORECASE) or re.search(
            entry["pattern"],
            compact,
            flags=re.IGNORECASE,
        ):
            return {
                "model_base": compact,
                **{col: entry[col] for col in ANNOTATION_COLUMNS},
            }

    return {
        "model_base": compact,
        "representation_family": "unknown",
        "input_modality": "unknown",
        "model_class": "unknown",
        "baseline_role": "needs manual annotation",
        "comparable_to_ours": "unknown",
        "notes": "No rule matched; inspect model name and add annotation rule.",
    }


def annotate_model_table(
    df: pd.DataFrame,
    *,
    model_col: str | None = None,
) -> pd.DataFrame:
    """Annotate raw Praski rows or Table 1/6-like summaries.

    Uses:
      - embedder column for raw Praski rows
      - Model column for summary tables
    """
    if model_col is None:
        if "embedder" in df.columns:
            model_col = "embedder"
        elif "Model" in df.columns:
            model_col = "Model"
        else:
            raise ValueError("Could not infer model column. Expected 'embedder' or 'Model'.")

    assert model_col is not None

    annotation_rows = [annotate_model_name(x) for x in df[model_col].astype(str)]
    annotation_cols = ["model_base", *ANNOTATION_COLUMNS]

    base_cols = list(df.columns)
    insert_at = base_cols.index(model_col) + 1

    ordered_data: dict[str, list[object]] = {}
    for col in base_cols[:insert_at]:
        ordered_data[col] = df[col].tolist()

    for col in annotation_cols:
        ordered_data[col] = [row[col] for row in annotation_rows]

    for col in base_cols[insert_at:]:
        ordered_data[col] = df[col].tolist()

    return pd.DataFrame(ordered_data, index=df.index)


def annotate_csv_or_tsv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    model_col: str | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() == ".tsv":
        df = pd.read_csv(input_path, sep="\t")
    else:
        df = pd.read_csv(input_path)

    annotated = annotate_model_table(df, model_col=model_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".tsv":
        annotated.to_csv(output_path, sep="\t", index=False)
    else:
        annotated.to_csv(output_path, index=False)


def summarize_model_categories(df: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_model_table(df)

    model_col = "embedder" if "embedder" in annotated.columns else "Model"

    return (
        annotated[
            [
                model_col,
                "model_base",
                "representation_family",
                "input_modality",
                "model_class",
                "baseline_role",
                "comparable_to_ours",
                "notes",
            ]
        ]
        .drop_duplicates()
        .sort_values(["comparable_to_ours", "representation_family", model_col])
        .reset_index(drop=True)
    )
