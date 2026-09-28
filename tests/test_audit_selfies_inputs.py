from audit_selfies_inputs import audit_strings


def test_audit_captures_failures_and_counts_bos_eos_for_truncation():
    counts, unknown = audit_strings(
        ["[C][O]", "[C].[O]", "[C]x[O]", "", None, "[C]" * 4],
        {"[C]": 5, "[O]": 6},
        max_length=5,
    )
    assert counts == {
        "n_inputs": 6,
        "n_disconnected": 1,
        "n_malformed": 1,
        "n_unknown": 2,
        "n_truncated_at_max_length": 1,
        "n_empty_or_nonstring": 2,
    }
    assert unknown == {".": 1}
