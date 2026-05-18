# upload_tokenizer.py  — run from repo root
import shutil
from pathlib import Path
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from huggingface_hub import HfApi

REPO_ID = "HauserGroup/ApeTokenizer-SELFIES"
TMP = Path("./tmp-hf-tokenizer")

# 1. Load your trained tokenizer from the custom json file
tokenizer = APEPreTrainedTokenizer(representation="SELFIES", model_max_length=256)
tokenizer.load_vocabulary_file("tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.json")

# 2. save_pretrained writes vocab.json, tokenizer_config.json,
#    special_tokens_map.json — with auto_map already wired in
tokenizer.save_pretrained(str(TMP))

# 3. tokenization_ape.py must live in the repo root on HF
#    so AutoTokenizer can find it via trust_remote_code
shutil.copy("src/modernmolbert/tokenization_ape.py", TMP / "tokenization_ape.py")

# 4. Upload the whole folder as a model repo

api = HfApi()


api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=True,
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(TMP),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add APE SELFIES tokenizer",
)

# 5. Clean up
shutil.rmtree(TMP)
print(f"Done — https://huggingface.co/{REPO_ID}")
