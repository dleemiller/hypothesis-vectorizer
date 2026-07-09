"""Guard against silent label mis-mapping in the hardcoded ClassLabel orderings.

banking77 / clinc150 / goemotions ship their class lists as literals in data.py (the label field
is an int index into the dataset's ClassLabel). If a literal's ORDER drifts from the dataset's, every
example is silently mislabeled and the pipeline still 'runs' — the worst kind of bug. These tests
assert the literal == the real HuggingFace ClassLabel names, exactly.

Marked `network` (hits HF) so the default suite stays fast/offline; run with `-m network`.
"""

import pytest

from hypothesis_vectorizer.train.data import _SPECS

_CLASSLABEL_DATASETS = ["banking77", "clinc150", "goemotions"]


@pytest.mark.network
@pytest.mark.parametrize("name", _CLASSLABEL_DATASETS)
def test_hardcoded_classlabel_order_matches_hf(name):
    from datasets import load_dataset

    spec = _SPECS[name]
    ds = load_dataset(spec["hf"], revision=spec.get("revision"), split="train")
    feature = ds.features[spec["label_field"]]
    hf_names = getattr(feature, "names", None)
    assert hf_names is not None, f"{name}: label field {spec['label_field']!r} is not a ClassLabel"
    assert spec["classes"] == hf_names, (
        f"{name}: hardcoded class order drifted from HuggingFace. "
        f"Regenerate from ds.features[{spec['label_field']!r}].names"
    )
    # descriptions stay aligned 1:1 with classes (the proposer relies on the pairing)
    assert len(spec["descriptions"]) == len(spec["classes"])
    for c, d in zip(spec["classes"], spec["descriptions"]):
        assert d.startswith(f"{c}:"), f"{name}: description for {c!r} must start with '{c}:'"
