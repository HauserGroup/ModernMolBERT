"""Unit test for HuggingFace model-card generation (`build_readme`).

Self-contained: builds minimal config / run-args / trainer-state files in a
tmp dir and checks the generated card has the expected frontmatter, metadata,
and the SELFIES encode + tokenize quickstart. No checkpoint, torch, or network.
"""

import json
from pathlib import Path

from modernmolbert.upload_model import build_readme

REPO_ID = "HauserGroup/ModernMolBERT-test"
VOCAB_SIZE = 631


def _write_checkpoint_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Create the minimal files build_readme reads; return (source_dir, run_dir)."""
    source_dir = tmp_path / "final_model"
    run_dir = tmp_path
    source_dir.mkdir()

    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "hidden_size": 512,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "max_position_embeddings": 128,
            }
        )
    )
    (run_dir / "run_args.json").write_text(
        json.dumps(
            {
                "dataset_name": "chembl36_selfies",
                "model_size": "small",
                "mlm_probability": 0.15,
                "masking_strategy": "standard",
                "learning_rate": 4e-4,
                "max_seq_length": 128,
            }
        )
    )
    # trainer_state is intentionally present to confirm no metrics leak into the card.
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"best_metric": 0.3744, "best_global_step": 30000})
    )
    return source_dir, run_dir


def test_build_readme_structure_and_quickstart(tmp_path: Path) -> None:
    source_dir, run_dir = _write_checkpoint_stub(tmp_path)

    card = build_readme(source_dir, run_dir, repo_id=REPO_ID, vocab_size=VOCAB_SIZE)

    # YAML frontmatter for a fill-mask transformers model
    assert card.startswith("---\n")
    assert "library_name: transformers" in card
    assert "pipeline_tag: fill-mask" in card
    assert "- selfies" in card

    # Title / identity
    assert f"# {REPO_ID}" in card

    # Config is reflected without pushing training-run internals above the quickstart.
    assert f"| vocab_size | {VOCAB_SIZE} |" in card
    assert "| hidden_size | 512 |" in card
    assert "## Training" not in card
    assert "| masking_strategy | standard |" not in card

    # The core ask: a minimal SELFIES tokenize example, with NO sf.encoder step
    assert "import selfies as sf" not in card
    assert "sf.encoder(" not in card
    assert "selfies = '[C][N][Branch1][C]" in card  # SELFIES string literal, not converted
    assert "subfolder='ape_tokenizer'" in card
    assert "AutoTokenizer.from_pretrained(\n    repo,\n    subfolder='ape_tokenizer'," in card
    assert "Current Transformers releases disable custom root tokenizers" in card
    assert "model = AutoModel.from_pretrained(repo).eval()" in card
    assert "mlm = AutoModelForMaskedLM.from_pretrained(repo)" in card
    assert REPO_ID in card  # repo id is substituted into the snippet
    assert card.index("## Model Details") < card.index("## How to Get Started")
    assert card.index("## How to Get Started") < card.index("## Uses")
    assert "frozen" not in card.lower()

    # The example prints labeled tokenizer output and embedding shape.
    assert "print(f\"Token IDs:\\n{inputs['input_ids'][0].tolist()}\\n\")" in card
    assert "tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])" in card
    assert "embedding_preview = [round(x, 4) for x in embedding[0, :5].tolist()]" in card
    assert 'print(f"Tokens:\\n{tokens}\\n")' in card
    assert "embedding = outputs.last_hidden_state[:, 0]" in card
    assert 'print(f"Embedding shape: {tuple(embedding.shape)}")' in card
    assert 'print(f"Embedding first 5 values:\\n{embedding_preview}")' in card
    assert 'print(f"Logits shape: {tuple(logits.shape)}")' in card
    assert "Output:\n\n```text\nToken IDs:" in card
    assert "Embedding first 5 values:" in card
    assert "Logits shape: (1, sequence_length, 631)" in card

    # Embedding example uses the [CLS]/first-token state.
    assert "last_hidden_state[:, 0]" in card

    # No evaluation metrics anywhere on the card
    assert "best_eval_loss" not in card
    assert "accuracy" not in card.lower()
    assert "eval_loss" not in card


def test_build_readme_handles_missing_optional_files(tmp_path: Path) -> None:
    """run_args / trainer_state are optional; card still generates from config alone."""
    source_dir = tmp_path / "final_model"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "modernbert", "hidden_size": 768})
    )

    card = build_readme(source_dir, tmp_path, repo_id=REPO_ID, vocab_size=VOCAB_SIZE)

    assert f"# {REPO_ID}" in card
    assert "print(f\"Token IDs:\\n{inputs['input_ids'][0].tolist()}\\n\")" in card
    assert 'print(f"Embedding shape: {tuple(embedding.shape)}")' in card
    assert "## Training" not in card
    assert "best_eval_loss" not in card
