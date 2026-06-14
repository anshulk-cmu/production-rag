import math

from evaluation.metrics import (
    compute_average_precision,
    compute_map,
    compute_metrics,
    compute_ndcg_at_k,
)

# Relevant docs land at ranks 1 and 3 out of four retrieved.
RETRIEVED = ["d1", "x", "d2", "y"]
RELEVANT = {"d1", "d2"}
EXPECTED_AP = (1.0 + 2 / 3) / 2  # precision 1/1 at rank 1, 2/3 at rank 3, over 2 relevant


def test_average_precision_hand_computed():
    assert math.isclose(compute_average_precision(RETRIEVED, RELEVANT), EXPECTED_AP, rel_tol=1e-9)


def test_average_precision_empty_relevant():
    assert compute_average_precision(RETRIEVED, set()) == 0.0


def test_map_is_average_precision_not_ndcg():
    result = compute_metrics(RETRIEVED, RELEVANT, k_values=[1, 4])
    ndcg = compute_ndcg_at_k(RETRIEVED, RELEVANT, 4)
    assert math.isclose(result.map_score, EXPECTED_AP, rel_tol=1e-9)
    # The old bug set MAP to NDCG@max_k; confirm they are now different.
    assert not math.isclose(result.map_score, ndcg, rel_tol=1e-6)


def test_single_query_map_matches_compute_map():
    result = compute_metrics(RETRIEVED, RELEVANT, k_values=[1, 4])
    assert math.isclose(result.map_score, compute_map([RETRIEVED], [RELEVANT]), rel_tol=1e-9)
