"""Freeze the upstream SMIRK 0.3.0 OpenSMILES tokenizer for the pilot campaign."""

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

from modernmolbert.tokenization.load import load_verified_tokenizer
from modernmolbert.utils import file_sha256, resolve_special_ids

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tokenizer/experimental_smirk_v1"
EXAMPLES = (
    "CCO",
    "Cl[Pt@SP1](Cl)([NH3])[NH3]",
    "[O-][99Tc](=O)(=O)=O.[Na+]",
)


def stage(output_dir: Path) -> None:
    if importlib.metadata.version("smirk") != "0.3.0":
        raise ValueError("This experiment requires smirk==0.3.0")
    from smirk import SmirkTokenizerFast

    tokenizer = SmirkTokenizerFast(template="[BOS] $0 [EOS]", representation="SMILES")
    special_ids = resolve_special_ids(tokenizer)
    if len(set(special_ids.values())) != len(special_ids):
        raise ValueError("SMIRK special token IDs are not unique")
    for smiles in EXAMPLES:
        ids = tokenizer(smiles)["input_ids"]
        if (
            ids[0] != special_ids["bos_token"]
            or ids[-1] != special_ids["eos_token"]
            or tokenizer.unk_token_id in ids
            or "".join(tokenizer.tokenize(smiles)) != smiles
            or tokenizer.decode(ids, skip_special_tokens=True) != smiles
        ):
            raise ValueError(f"SMIRK failed round-trip preflight: {smiles}")

    with tempfile.TemporaryDirectory() as temporary:
        tokenizer.save_pretrained(temporary)
        payload = (Path(temporary) / "tokenizer.json").read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "smirk_smiles.json"
    if tokenizer_path.exists() and tokenizer_path.read_bytes() != payload:
        raise FileExistsError(f"Frozen SMIRK tokenizer differs: {tokenizer_path}")
    tokenizer_path.write_bytes(payload)
    metadata = {
        "algorithm": "SMIRK",
        "representation": "SMILES",
        "source": "https://eeg.engin.umich.edu/smirk/",
        "smirk_version": "0.3.0",
        "template": "[BOS] $0 [EOS]",
        "special_ids": special_ids,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "vocab_size": len(tokenizer),
    }
    metadata_path = output_dir / "smirk_smiles.metadata.json"
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if metadata_path.exists() and metadata_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Frozen SMIRK metadata differs: {metadata_path}")
    metadata_path.write_text(encoded, encoding="utf-8")
    reloaded, _, _, _ = load_verified_tokenizer(tokenizer_path, metadata_path)
    for smiles in EXAMPLES:
        if reloaded(smiles)["input_ids"] != tokenizer(smiles)["input_ids"]:
            raise ValueError(f"SMIRK tokenizer reload changed {smiles}")
    print(f"Staged {tokenizer_path} ({metadata['tokenizer_sha256']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stage(args.output_dir)


if __name__ == "__main__":
    main()
