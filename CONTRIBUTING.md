# Contributing

ModernMolBERT evaluates molecular representations through a shared frozen-feature benchmark pipeline. Contributions should preserve comparability across datasets, featurizers, downstream models, metrics, and plots.

This document currently focuses on **evaluation dataset contributions**. Featurizer and evaluator contribution guidelines can be added as separate sections later.

---

## Scope

The benchmark stack currently supports:

```text
regression
binary classification
```

Do not add dataset-specific benchmark runners, dataset-specific downstream models, or one-off metric code inside dataset loaders.

Contributions should be:

- reproducible,
- testable,
- documented,
- compatible with the shared evaluation pipeline,
- narrow enough to review.

---

## Files involved

A dataset contribution usually changes only these files:

```text
src/modernmolbert/eval/contributed_datasets.py
tests/test_eval_<dataset_name>.py
configs/eval_suites/<suite_name>.yaml
```

The registry machinery lives in:

```text
src/modernmolbert/eval/dataset_registry.py
```

Most dataset contributors should not need to edit `dataset_registry.py`.

Do **not** add large raw data files to the repository unless explicitly approved.

---

## Dataset contribution summary

A contributed dataset must provide:

```text
1. a loader function in src/modernmolbert/eval/contributed_datasets.py
2. a DatasetSpec registration in register_contributed_datasets()
3. one small loader test
4. an optional suite YAML reference
```

Datasets are contributed **through code**, not by manually adding standalone YAML files or local raw CSV files to the repository.

The data itself should be hosted somewhere stable and accessible, or generated from a documented source. The `root` path in examples refers to a local cache or prepared-data directory, not data committed to the repository. If the data are hosted on Hugging Face, the loader should use `datasets.load_dataset(...)`; if the data require downloading/preparation, add a small script under `scripts/data_prep/` and document it. Dataset metadata belongs in the `DatasetSpec` and in the returned `EvalDataset.metadata`.

---

## The `EvalDataset` object

Every contributed dataset loader must return an `EvalDataset`.

`EvalDataset` is defined in:

```text
src/modernmolbert/eval/datasets.py
```

The object has this shape:

```python
@dataclass(frozen=True)
class EvalDataset:
    name: str
    task_type: Literal["classification", "regression"]
    task_names: list[str]
    train: pd.DataFrame
    valid: pd.DataFrame | None
    test: pd.DataFrame
    smiles_column: str = "smiles"
    selfies_column: str = "selfies"
    metadata: dict[str, Any] = field(default_factory=dict)
```

Construct it with:

```python
make_eval_dataset_from_splits(...)
```

Do not subclass `EvalDataset`.

---

## Required data shape

The source data should provide fixed splits:

```text
train
test
valid  # optional
```

Each split must contain one molecule column and one label column.

Binary classification example:

```csv
smiles,active
CCO,0
CCN,1
CCC,
```

Regression example:

```csv
smiles,pIC50
CCO,4.32
CCN,6.10
CCC,
```

Missing labels may exist in the source data, but the loader must drop them before returning `EvalDataset`.

Returned `EvalDataset` objects should not contain missing labels.

The default molecule column is:

```text
smiles
```

The label columns are the `task_names`.

---

## Minimal binary classification example

Suppose the dataset is called `my_activity`.

Expected loader input:

```text
train: smiles, active
test:  smiles, active
valid: smiles, active  # optional
```

Expected loader output:

```text
EvalDataset
  name: my_activity
  task_type: classification
  task_names: [active]
  train/test/valid: no missing active labels
  active labels: 0 or 1 only
```

Add the loader to:

```text
src/modernmolbert/eval/contributed_datasets.py
```

```python
from pathlib import Path

import pandas as pd

from modernmolbert.eval.dataset_registry import DatasetSpec, register_dataset
from modernmolbert.eval.datasets import EvalDataset, make_eval_dataset_from_splits


def load_my_activity(*, root: str | Path) -> EvalDataset:
    root = Path(root)

    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")

    cleaned = []
    for split_name, frame in [("train", train), ("test", test)]:
        frame = frame.copy()
        frame["active"] = pd.to_numeric(frame["active"], errors="coerce")
        frame = frame.dropna(subset=["active"]).reset_index(drop=True)

        invalid = set(frame["active"].astype(int).unique()) - {0, 1}
        if invalid:
            raise ValueError(
                f"{split_name} split contains non-binary labels: {sorted(invalid)}"
            )

        cleaned.append(frame)

    train, test = cleaned

    return make_eval_dataset_from_splits(
        name="my_activity",
        task_type="classification",
        task_names="active",
        train=train,
        test=test,
        smiles_column="smiles",
        metadata={
            "source": "My Activity Dataset",
            "source_url": "https://example.org/my_activity",
            "license": "CC-BY-4.0",
            "citation": "Example et al. 2026",
            "split_source": "published_split_v1",
            "label_definition": "active: 0=inactive, 1=active",
            "missing_label_policy": "Rows with missing active labels are dropped.",
        },
    )
```

---

## Register the dataset

In `register_contributed_datasets()`, add:

```python
register_dataset(
    DatasetSpec(
        name="my_activity",
        task_type="classification",
        task_names=("active",),
        loader=load_my_activity,
        description="Binary activity prediction benchmark.",
        source="https://example.org/my_activity",
        citation="Example et al. 2026",
        license="CC-BY-4.0",
    )
)
```

The registration is the contribution. A benchmark suite can reference the registered dataset, but should not define the dataset logic.

---

## Add one minimal test

Add:

```text
tests/test_eval_my_activity.py
```

```python
from pathlib import Path

import pandas as pd

from modernmolbert.eval.contributed_datasets import load_my_activity


def test_load_my_activity(tmp_path: Path) -> None:
    root = tmp_path / "my_activity"
    root.mkdir()

    pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC"], "active": [0, 1, None]}
    ).to_csv(root / "train.csv", index=False)

    pd.DataFrame(
        {"smiles": ["CCCl", "CCBr"], "active": [0, 1]}
    ).to_csv(root / "test.csv", index=False)

    dataset = load_my_activity(root=root)

    dataset.check()
    assert dataset.name == "my_activity"
    assert len(dataset.train) == 2
```

Run:

```bash
uv run pytest tests/test_eval_my_activity.py -q
```

---

## Reference it from a suite config

Create or edit a suite file under:

```text
configs/eval_suites/
```

For example:

```text
configs/eval_suites/core_contributed.yaml
```

Add:

```yaml
datasets:
  - loader: registered
    name: my_activity
    root: data/eval/my_activity  # local cache/prepared-data directory
```

The suite config references the registered dataset. It should not define dataset loading logic.
The loader is responsible for creating or reading this local prepared-data directory from the documented hosted source.

---

## Regression datasets

For regression, the structure is the same, but use:

```python
task_type="regression"
task_names="pIC50"
```

The label column must be numeric:

```python
frame["pIC50"] = pd.to_numeric(frame["pIC50"], errors="coerce")
frame = frame.dropna(subset=["pIC50"]).reset_index(drop=True)
```

Regression labels do not need binary validation.

---

## Correct reference example

A correct reference example lives in:

```text
src/modernmolbert/eval/contributed_datasets.py
```

Look at:

```python
load_example_activity_dataset(...)
```

It demonstrates:

```text
- expected train/valid/test file layout
- required columns
- dropping missing labels
- validating binary labels
- constructing EvalDataset with make_eval_dataset_from_splits(...)
- recording useful metadata
- the DatasetSpec registration pattern
```

The example is not registered as a real dataset by default. It is there to show the required style.

## Dataset properties

 - No larger than a couple of 1000 molecules
 - Binary classification datasets should not be extremely imbalanced. As a rough guideline, the minority class should normally be at least 10% of the labeled rows. More imbalanced datasets may still be useful, but they should be discussed before contribution because they may need special evaluation treatment.
 - Suitable for benchmarking i.e. not impossible for an featurizer e.g. no harder than a Random Forest works
 - For regression, approximately normally distributed e.g. pEC50 and not raw EC50
 - Clearly cited and appropriate license
 - Do not implement datasets from MoleculeNet

## Checklist

- [ ] Loader lives in `src/modernmolbert/eval/contributed_datasets.py`.
- [ ] Loader returns `EvalDataset`.
- [ ] Dataset is registered with `DatasetSpec`.
- [ ] Task is regression or binary classification.
- [ ] Binary labels are `0/1`.
- [ ] Regression labels are numeric.
- [ ] Missing labels are dropped before returning `EvalDataset`.
- [ ] Source URL, license, citation, split source, and label definition are recorded.
- [ ] One minimal test is included.
- [ ] Optional suite YAML references the registered dataset with `loader: registered`.

---

## What not to do

 - Do not contribute a reusable dataset only through YAML.

 - Do not add a new benchmark runner for one dataset.

 - Do not add dataset-specific featurization code.

 - Do not compute metrics in a dataset loader.

 - Do not train downstream models inside a dataset loader.

 - Do not randomly split data inside the suite runner.

 - Do not rely on task weights.

 - Do not return NaN labels inside `EvalDataset`.

 - Do not contribute multiclass datasets unless the benchmark stack has first been extended to support multiclass classification.

 - Do not commit large raw datasets unless explicitly approved.
