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


def test_audit_tallies_token_lengths_including_bos_eos():
    from collections import Counter

    from audit_selfies_inputs import length_summary

    lengths = Counter()
    audit_strings(["[C][O]", "[C]", "[C][C][C]"], {"[C]": 5, "[O]": 6}, lengths=lengths)
    assert lengths == {4: 1, 3: 1, 5: 1}
    assert length_summary(lengths) == {"n": 3, "mean": 4.0, "max": 5, "p50": 4, "p95": 5, "p99": 5}
    assert length_summary(Counter()) == {}
