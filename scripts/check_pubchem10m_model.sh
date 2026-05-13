#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-runs/pubchem10m_mps_base_overnight/final_model}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}"
  exit 1
fi

echo "Checking model reload and encoder output..."
echo "Model directory: ${MODEL_DIR}"

uv run python - <<PY
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import torch

model_dir = "${MODEL_DIR}"

mlm = AutoModelForMaskedLM.from_pretrained(model_dir)
enc = AutoModel.from_pretrained(model_dir)

tok = AutoTokenizer.from_pretrained(
    f"{model_dir}/ape_tokenizer",
    trust_remote_code=True,
)

batch = tok("[C][C][O]", add_special_tokens=True, return_tensors="pt")
batch = {
    key: value.unsqueeze(0) if hasattr(value, "ndim") and value.ndim == 1 else value
    for key, value in batch.items()
}

with torch.no_grad():
    mlm_out = mlm(**batch)
    enc_out = enc(**batch)

ids = batch["input_ids"][0].tolist()

print("ids:", ids)
print("tokens:", tok.convert_ids_to_tokens(ids))
print("mlm logits:", tuple(mlm_out.logits.shape))
print("encoder hidden:", tuple(enc_out.last_hidden_state.shape))
print("mlm finite:", torch.isfinite(mlm_out.logits).all().item())
print("encoder finite:", torch.isfinite(enc_out.last_hidden_state).all().item())

assert torch.isfinite(mlm_out.logits).all()
assert torch.isfinite(enc_out.last_hidden_state).all()
print("reload ok")
PY
