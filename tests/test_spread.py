from eval.metrics import InterviewMetrics
from eval.run import spread


def metrics(persona_id: str, recall: float, *, degenerate: bool = False) -> InterviewMetrics:
    return InterviewMetrics(
        persona_id=persona_id,
        stop_reason="coverage",
        turn_count=6,
        turns_to_coverage=6,
        hidden_fact_recall=recall,
        fact_scores=[],
        leaked_questions=[],
        forced_fallbacks=6 if degenerate else 0,
        distinct_question_ratio=0.5 if degenerate else 1.0,
        goals_covered=4,
        goals_total=4,
    )


def test_repeats_are_grouped_per_persona():
    out = spread([metrics("mira", 0.4), metrics("mira", 0.2), metrics("devan", 0.6)])

    assert "mira" in out
    assert "devan" in out
    assert "overall mean" in out


def test_a_wide_spread_warns_against_single_run_comparisons():
    out = spread([metrics("mira", 0.1), metrics("mira", 0.6)])

    assert "spread is wide" in out
    assert "0.10" in out
    assert "0.60" in out


def test_a_tight_spread_does_not_warn():
    out = spread([metrics("mira", 0.40), metrics("mira", 0.45)])

    assert "spread is wide" not in out


def test_degenerate_runs_are_excluded_from_the_mean():
    out = spread(
        [
            metrics("mira", 0.4),
            metrics("mira", 1.0, degenerate=True),
            metrics("mira", 0.4),
        ]
    )

    assert "2 valid runs" in out
    assert "overall mean 0.40" in out


def test_a_persona_with_no_valid_runs_is_shown_rather_than_dropped():
    out = spread([metrics("mira", 0.4), metrics("hanna", 1.0, degenerate=True)])

    assert "hanna" in out


def test_a_single_valid_run_refuses_to_report_spread():
    out = spread([metrics("mira", 0.4), metrics("hanna", 1.0, degenerate=True)])

    assert "too few valid runs" in out
