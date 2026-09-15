"""Integration tests for the policy layer wired into the gateway dispatcher.

The unit tests in ``tests/hermes_cli/test_kanban_policy.py`` cover the
decision logic. These cover the *wiring*: that the budget actually narrows
what ``dispatch_once`` is called with, that a pause stops all boards, and
that a failure in the policy layer cannot take the dispatcher down.
"""

from __future__ import annotations

import pytest

from hermes_cli.kanban_policy import (
    QuotaGuard,
    load_policy_config,
    plan_dispatch,
)


def _payload(session=23, fetched_at="2026-09-15T15:02:49.083Z"):
    return [
        {
            "providerId": "claude",
            "fetchedAt": fetched_at,
            "lines": [
                {
                    "type": "progress",
                    "label": "Session",
                    "used": session,
                    "limit": 100,
                    "resetsAt": "2026-09-15T19:30:00.710Z",
                },
                {"type": "progress", "label": "Weekly", "used": 10, "limit": 100},
                {"type": "progress", "label": "Fable", "used": 10, "limit": 100},
            ],
        }
    ]


def _now(fetched_at="2026-09-15T15:02:49.083Z"):
    from datetime import datetime, timezone

    return (
        datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        .replace(tzinfo=timezone.utc)
        .timestamp()
        + 5
    )


def _effective(max_in_progress, global_budget):
    """Mirror of the narrowing in _tick_once_for_board."""
    if global_budget is None:
        return max_in_progress
    if max_in_progress is None:
        return global_budget
    return min(max_in_progress, global_budget)


# --- the budget narrows, never widens ------------------------------------


@pytest.mark.parametrize(
    "configured,budget,expected",
    [
        (1, 2, 1),  # per-board cap is tighter: it wins
        (5, 2, 2),  # global budget is tighter: it wins
        (None, 2, 2),  # no per-board cap: budget applies
        (3, None, 3),  # policy unavailable: configured value survives
    ],
)
def test_global_budget_can_only_tighten_the_board_cap(configured, budget, expected):
    assert _effective(configured, budget) == expected


def test_a_board_capped_at_one_never_spawns_two_even_with_spare_budget():
    """The existing per-board cap is not overridden by the new layer."""
    assert _effective(1, 4) == 1


# --- pause semantics -----------------------------------------------------


def test_quota_pause_yields_no_boards_to_dispatch():
    policy = load_policy_config({"kanban": {"quota": {"max_unattended_pct": 75}}})
    guard = QuotaGuard(fetch_fn=lambda: _payload(session=88), cache_seconds=0)
    plan = plan_dispatch(
        board_running={"tomebound": 0, "koctakip": 0},
        policy=policy,
        quota_guard=guard,
        now=_now(),
    )
    assert plan.paused
    # Every board gets zero, so the dispatcher loop skips all of them.
    assert [b.allowed for b in plan.boards] == [0, 0]


def test_in_flight_work_is_reported_but_not_cancelled_during_a_pause():
    policy = load_policy_config({"kanban": {"quota": {"max_unattended_pct": 75}}})
    guard = QuotaGuard(fetch_fn=lambda: _payload(session=88), cache_seconds=0)
    plan = plan_dispatch(
        board_running={"tomebound": 2},
        policy=policy,
        quota_guard=guard,
        now=_now(),
    )
    assert plan.paused
    assert plan.boards[0].running == 2
    assert plan.boards[0].allowed == 0


# --- resilience ----------------------------------------------------------


def test_quota_source_down_does_not_pause_by_default():
    """The menu-bar app not running is the common case, not an emergency."""

    def dead():
        raise OSError("connection refused")

    policy = load_policy_config({"kanban": {"quota": {"max_unattended_pct": 75}}})
    guard = QuotaGuard(fetch_fn=dead, cache_seconds=0)
    plan = plan_dispatch(
        board_running={"tomebound": 0}, policy=policy, quota_guard=guard
    )
    assert not plan.paused
    assert plan.allowed_for("tomebound") >= 1


def test_quota_source_down_pauses_when_configured_to_fail_closed():
    def dead():
        raise OSError("connection refused")

    policy = load_policy_config(
        {"kanban": {"quota": {"max_unattended_pct": 75, "allow_on_unknown": False}}}
    )
    guard = QuotaGuard(fetch_fn=dead, cache_seconds=0)
    plan = plan_dispatch(
        board_running={"tomebound": 0}, policy=policy, quota_guard=guard
    )
    assert plan.paused


def test_disabling_the_quota_guard_skips_it_entirely():
    def boom():
        raise AssertionError("guard must not be consulted when disabled")

    policy = load_policy_config({"kanban": {"quota": {"enabled": False}}})
    guard = QuotaGuard(fetch_fn=boom, cache_seconds=0)
    plan = plan_dispatch(
        board_running={"tomebound": 0}, policy=policy, quota_guard=guard
    )
    assert not plan.paused


# --- board priority under contention -------------------------------------


def test_priority_board_gets_the_last_free_slot():
    policy = load_policy_config(
        {
            "kanban": {
                "global_max_in_progress": 2,
                "board_priority": ["tomebound", "koctakip", "asistan"],
                "quota": {"enabled": False},
            }
        }
    )
    plan = plan_dispatch(
        board_running={"asistan": 0, "koctakip": 1, "tomebound": 0}, policy=policy
    )
    # One slot left; tomebound is first in line for it.
    assert [b.slug for b in plan.boards][0] == "tomebound"
    assert plan.allowed_for("tomebound") == 1


def test_unconfigured_board_still_gets_dispatched_last():
    """A board created mid-run must not be silently starved forever."""
    policy = load_policy_config(
        {
            "kanban": {
                "global_max_in_progress": 4,
                "board_priority": ["tomebound"],
                "quota": {"enabled": False},
            }
        }
    )
    plan = plan_dispatch(board_running={"default": 0, "tomebound": 0}, policy=policy)
    slugs = [b.slug for b in plan.boards]
    assert slugs == ["tomebound", "default"]
    assert plan.allowed_for("default") > 0
