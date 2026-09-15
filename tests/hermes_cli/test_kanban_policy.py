"""Tests for the unattended-dispatch policy layer.

These assert the observable decision — how many workers a board may start,
and whether the factory is paused — rather than the internals of how the
budget is computed. The fixtures for the quota reader are real OpenUsage
payload shapes, captured from the live endpoint on 2026-09-15.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli.kanban_policy import (
    DEFAULT_MAX_UNATTENDED_PCT,
    QuotaGuard,
    _parse_openusage,
    count_running_readonly,
    load_policy_config,
    order_boards,
    plan_dispatch,
    runtime_limits_for_role,
)


def _usage_payload(
    *, session=23, weekly=13, fable=16, fetched_at="2026-09-15T15:02:49.083Z"
):
    """A payload shaped exactly like OpenUsage /v1/usage returns."""
    return [
        {
            "plan": "Max 5x",
            "displayName": "Claude",
            "fetchedAt": fetched_at,
            "providerId": "claude",
            "lines": [
                {
                    "resetsAt": "2026-09-15T19:30:00.710Z",
                    "type": "progress",
                    "label": "Session",
                    "format": {"kind": "percent"},
                    "used": session,
                    "limit": 100,
                    "periodDurationMs": 18000000,
                },
                {
                    "resetsAt": "2026-09-15T20:00:00.710Z",
                    "type": "progress",
                    "label": "Weekly",
                    "format": {"kind": "percent"},
                    "used": weekly,
                    "limit": 100,
                    "periodDurationMs": 604800000,
                },
                {
                    "resetsAt": "2026-09-15T20:00:00.710Z",
                    "type": "progress",
                    "label": "Fable",
                    "format": {"kind": "percent"},
                    "used": fable,
                    "limit": 100,
                    "periodDurationMs": 604800000,
                },
                {"label": "Today", "type": "text", "value": "$12.80 · 14.1M tokens"},
            ],
        }
    ]


def _fresh_now(fetched_at="2026-09-15T15:02:49.083Z"):
    """A 'now' a few seconds after the given fetch time."""
    from datetime import datetime, timezone

    ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc
    )
    return ts.timestamp() + 5


def _guard(payload, **kw):
    return QuotaGuard(fetch_fn=lambda: payload, cache_seconds=0, **kw)


# --- quota reading -------------------------------------------------------


def test_parses_the_three_progress_lines():
    r = _parse_openusage(_usage_payload(), now=_fresh_now())
    assert r.ok
    assert r.window_pct == 23
    assert r.weekly_pct == 13
    assert r.weekly_fable_pct == 16
    assert r.worst_pct == 23


def test_worst_pct_tracks_the_closest_limit_not_the_weekly_one():
    """Being under the weekly cap is no comfort when the window is full."""
    r = _parse_openusage(_usage_payload(session=91, weekly=10, fable=12), now=_fresh_now())
    assert r.worst_pct == 91


def test_missing_claude_provider_is_not_ok():
    assert not _parse_openusage([{"providerId": "codex", "lines": []}]).ok


def test_garbage_payload_is_not_ok():
    assert not _parse_openusage({"nope": True}).ok
    assert not _parse_openusage([]).ok


def test_fetch_failure_does_not_raise():
    def boom():
        raise OSError("connection refused")

    g = QuotaGuard(fetch_fn=boom, cache_seconds=0)
    r = g.read()
    assert not r.ok
    assert "fetch failed" in r.reason


# --- the staleness trap --------------------------------------------------


def test_stale_reading_is_treated_as_unknown_not_as_under_threshold():
    """The live endpoint served a 2.5h-old snapshot; trusting it blindly
    would wave work through all evening on a morning number."""
    payload = _usage_payload(session=23)
    stale_now = _fresh_now() + 9000  # 2.5 hours later
    g = _guard(payload, max_age_seconds=1800)
    r = g.read(now=stale_now)
    assert not r.ok
    assert "stale" in r.reason


def test_stale_reading_holds_when_unknown_is_not_allowed():
    g = _guard(_usage_payload(session=23), max_age_seconds=1800)
    may, reason = g.evaluate(
        threshold_pct=75, allow_on_unknown=False, now=_fresh_now() + 9000
    )
    assert may is False
    assert "stale" in reason


def test_fresh_reading_under_threshold_permits_claims():
    g = _guard(_usage_payload(session=23))
    may, _ = g.evaluate(threshold_pct=75, allow_on_unknown=True, now=_fresh_now())
    assert may is True


# --- threshold maths -----------------------------------------------------


@pytest.mark.parametrize(
    "session,expected",
    [(74, True), (75, False), (76, False), (99, False)],
)
def test_threshold_is_inclusive_at_the_boundary(session, expected):
    g = _guard(_usage_payload(session=session))
    may, _ = g.evaluate(threshold_pct=75, allow_on_unknown=True, now=_fresh_now())
    assert may is expected


def test_fable_line_alone_can_trip_the_guard():
    """D10 reserves a quarter of the weekly Fable allowance specifically."""
    g = _guard(_usage_payload(session=10, weekly=10, fable=80))
    may, reason = g.evaluate(threshold_pct=75, allow_on_unknown=True, now=_fresh_now())
    assert may is False
    assert "fable=80" in reason


def test_unknown_quota_proceeds_when_configured_to():
    g = QuotaGuard(fetch_fn=lambda: (_ for _ in ()).throw(OSError()), cache_seconds=0)
    may, _ = g.evaluate(threshold_pct=75, allow_on_unknown=True)
    assert may is True


# --- caching -------------------------------------------------------------


def test_reading_is_cached_for_the_tick():
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return _usage_payload()

    g = QuotaGuard(fetch_fn=counting, cache_seconds=60)
    base = _fresh_now()
    g.read(now=base)
    g.read(now=base + 1)
    g.read(now=base + 59)
    assert calls["n"] == 1
    g.read(now=base + 61)
    assert calls["n"] == 2


# --- board ordering ------------------------------------------------------


def test_boards_are_served_in_configured_priority_order():
    assert order_boards(
        ["asistan", "koctakip", "tomebound"], ["tomebound", "koctakip", "asistan"]
    ) == ["tomebound", "koctakip", "asistan"]


def test_unlisted_board_sorts_last_but_is_never_dropped():
    out = order_boards(["newproj", "tomebound"], ["tomebound", "koctakip"])
    assert out == ["tomebound", "newproj"]


# --- the global cap ------------------------------------------------------


def _policy(**over):
    base = load_policy_config(
        {
            "kanban": {
                "global_max_in_progress": 2,
                "board_priority": ["tomebound", "koctakip", "asistan"],
                "quota": {"enabled": False},
            }
        }
    )
    base.update(over)
    return base


def test_global_cap_counts_across_boards_not_per_board():
    """Three boards each under their own cap must not run six workers."""
    plan = plan_dispatch(
        board_running={"tomebound": 1, "koctakip": 1, "asistan": 0}, policy=_policy()
    )
    assert plan.global_running == 2
    assert all(b.allowed == 0 for b in plan.boards)


def test_free_slots_go_to_the_highest_priority_board_first():
    plan = plan_dispatch(
        board_running={"asistan": 0, "koctakip": 0, "tomebound": 0}, policy=_policy()
    )
    assert [b.slug for b in plan.boards] == ["tomebound", "koctakip", "asistan"]
    assert plan.boards[0].allowed == 2


def test_partial_budget_is_offered_to_each_board_in_order():
    plan = plan_dispatch(
        board_running={"tomebound": 1, "koctakip": 0, "asistan": 0}, policy=_policy()
    )
    assert plan.allowed_for("tomebound") == 1
    assert plan.allowed_for("koctakip") == 1


def test_running_over_cap_yields_no_spawns_and_does_not_go_negative():
    plan = plan_dispatch(board_running={"tomebound": 5}, policy=_policy())
    assert plan.allowed_for("tomebound") == 0


# --- quota pausing the whole factory -------------------------------------


def test_threshold_breach_pauses_every_board():
    policy = load_policy_config(
        {
            "kanban": {
                "global_max_in_progress": 2,
                "board_priority": ["tomebound", "koctakip", "asistan"],
                "quota": {"max_unattended_pct": 75},
            }
        }
    )
    guard = _guard(_usage_payload(session=80))
    plan = plan_dispatch(
        board_running={"tomebound": 0, "koctakip": 0, "asistan": 0},
        policy=policy,
        quota_guard=guard,
        now=_fresh_now(),
    )
    assert plan.paused
    assert all(b.allowed == 0 for b in plan.boards)
    assert "80" in plan.pause_reason


def test_pause_does_not_disturb_running_work():
    """D10: running cards finish; only new claims stop."""
    policy = load_policy_config({"kanban": {"quota": {"max_unattended_pct": 75}}})
    guard = _guard(_usage_payload(session=90))
    plan = plan_dispatch(
        board_running={"tomebound": 2}, policy=policy, quota_guard=guard, now=_fresh_now()
    )
    assert plan.paused
    assert plan.boards[0].running == 2  # still reported, untouched
    assert plan.boards[0].allowed == 0


def test_under_threshold_does_not_pause():
    policy = load_policy_config({"kanban": {"quota": {"max_unattended_pct": 75}}})
    guard = _guard(_usage_payload(session=23))
    plan = plan_dispatch(
        board_running={"tomebound": 0}, policy=policy, quota_guard=guard, now=_fresh_now()
    )
    assert not plan.paused


# --- config parsing ------------------------------------------------------


def test_defaults_apply_to_an_empty_config():
    p = load_policy_config({})
    assert p["global_max_in_progress"] == 2
    assert p["quota_threshold_pct"] == DEFAULT_MAX_UNATTENDED_PCT
    assert p["board_priority"] == []


def test_malformed_values_fall_back_rather_than_raising():
    p = load_policy_config(
        {
            "kanban": {
                "global_max_in_progress": "banana",
                "board_priority": "not-a-list",
                "quota": {"max_unattended_pct": 999},
            }
        }
    )
    assert p["global_max_in_progress"] == 2
    assert p["board_priority"] == []
    assert p["quota_threshold_pct"] == DEFAULT_MAX_UNATTENDED_PCT


def test_none_config_is_survivable():
    assert load_policy_config(None)["global_max_in_progress"] == 2


# --- runtime limits ------------------------------------------------------


@pytest.mark.parametrize(
    "assignee,soft",
    [("tb-coder", 7200), ("kt-tester", 9000), ("ap-reviewer", 5400)],
)
def test_runtime_table_matches_role_across_project_prefixes(assignee, soft):
    policy = load_policy_config({})
    assert runtime_limits_for_role(assignee, policy)[0] == soft


def test_hard_limit_is_twice_the_soft_limit():
    policy = load_policy_config({})
    s, h = runtime_limits_for_role("tb-coder", policy)
    assert h == s * 2


def test_unknown_role_gets_the_most_generous_soft_limit():
    policy = load_policy_config({})
    s, _ = runtime_limits_for_role("tomebound-lead", policy)
    assert s == 9000


# --- the running-count probe must not write -----------------------------


def _make_board(tmp_path, running=0, done=0):
    import sqlite3

    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)")
    for i in range(running):
        conn.execute("INSERT INTO tasks VALUES (?,'running')", (f"r{i}",))
    for i in range(done):
        conn.execute("INSERT INTO tasks VALUES (?,'done')", (f"d{i}",))
    conn.commit()
    conn.close()
    return db


def test_readonly_probe_counts_only_running(tmp_path):
    assert count_running_readonly(_make_board(tmp_path, running=2, done=5)) == 2


def test_readonly_probe_never_creates_a_missing_db(tmp_path):
    """The dispatcher probes every board every tick; it must not
    materialise a DB for a board that does not exist."""
    missing = tmp_path / "nope" / "kanban.db"
    assert count_running_readonly(missing) == 0
    assert not missing.exists()


def test_readonly_probe_cannot_write(tmp_path):
    """Opening ?mode=ro means a schema migration can never run here."""
    import sqlite3

    db = _make_board(tmp_path, running=1)
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO tasks VALUES ('x','running')")
    finally:
        conn.close()


def test_readonly_probe_treats_a_corrupt_db_as_idle(tmp_path):
    """A quarantined/corrupt board must count as 0, not raise — the
    dispatcher disables those boards and must not be taken down by one."""
    db = tmp_path / "kanban.db"
    db.write_text("not sqlite", encoding="utf-8")
    assert count_running_readonly(db) == 0


def test_readonly_probe_treats_legacy_schema_as_idle(tmp_path):
    import sqlite3

    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    assert count_running_readonly(db) == 0


def test_runtime_table_is_overridable_from_config():
    policy = load_policy_config(
        {"kanban": {"runtime": {"soft_seconds": {"coder": 60}, "hard_multiplier": 3}}}
    )
    assert runtime_limits_for_role("tb-coder", policy) == (60, 180)
