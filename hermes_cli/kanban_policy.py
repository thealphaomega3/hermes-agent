"""Unattended-dispatch policy for the multi-board factory.

The dispatcher in :mod:`hermes_cli.kanban_db` decides *whether a given board
can spawn one more worker*. It has no notion of the three things an
unattended, multi-board factory needs:

1. a **global** concurrency cap across every board (``dispatch_once`` counts
   per board, so three boards with a per-board cap of 2 run six workers),
2. a **fixed board order** so the boards that matter most are served first
   when that global cap is the binding constraint,
3. a **quota guard** so overnight work cannot spend the whole weekly
   allowance and lock the user out of their own account.

This module is the policy layer the dispatcher consults before claiming.
It deliberately owns no dispatch mechanics: it answers "how many workers may
board X start right now, if any" and the caller does the spawning. That keeps
the existing per-board ``max_in_progress`` / ``max_in_progress_per_profile``
caps working underneath, unchanged.

Every value is read from the ``kanban`` block of config on each call rather
than captured once, matching the existing ``auto_decompose`` behaviour: an
operator who trips a threshold and wants the factory paused NOW must not have
to restart the gateway for it to take effect.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# OpenUsage.app publishes Claude plan usage on a loopback port. It is the only
# source on this host that reports the weekly *Fable* allowance separately from
# the overall weekly figure, which is what the quota decision is actually about.
DEFAULT_QUOTA_URL = "http://127.0.0.1:6736/v1/usage"

# The endpoint serves a cached snapshot and does NOT refresh on read: observed
# serving a reading 2.5 hours old. A stale number is worse than no number,
# because "23% used" from this morning waves work through all evening. Past
# this age the reading is treated as unknown.
DEFAULT_QUOTA_MAX_AGE_SECONDS = 1800

DEFAULT_GLOBAL_MAX_IN_PROGRESS = 2
DEFAULT_MAX_UNATTENDED_PCT = 75

# Per-role soft runtime limits (seconds). At the soft limit the lead is asked
# what to do; the hard limit is a multiple of it and is not negotiable.
DEFAULT_RUNTIME_SOFT_SECONDS = {"coder": 7200, "tester": 9000, "reviewer": 5400}
DEFAULT_RUNTIME_HARD_MULTIPLIER = 2


@dataclass(frozen=True)
class QuotaReading:
    """One usage snapshot, normalised to percentages.

    ``ok`` is False when the reading could not be obtained or is too old to
    act on. Callers must treat ``ok=False`` as *unknown*, never as *under
    threshold* — see :meth:`QuotaGuard.evaluate`.
    """

    ok: bool
    window_pct: Optional[float] = None
    weekly_pct: Optional[float] = None
    weekly_fable_pct: Optional[float] = None
    resets_at: Optional[str] = None
    age_seconds: Optional[float] = None
    reason: str = ""

    @property
    def worst_pct(self) -> Optional[float]:
        """Highest of the tracked percentages, or None when unknown.

        The guard trips on whichever limit is closest to its ceiling: being
        under the weekly cap is no comfort when the 5-hour window is full.
        """
        vals = [
            v
            for v in (self.window_pct, self.weekly_pct, self.weekly_fable_pct)
            if isinstance(v, (int, float))
        ]
        return max(vals) if vals else None


def _parse_openusage(payload: object, *, now: Optional[float] = None) -> QuotaReading:
    """Normalise an OpenUsage ``/v1/usage`` body into a QuotaReading.

    Split out from the fetch so the threshold maths can be tested against
    fixtures without a live app on the port.
    """
    now = time.time() if now is None else now
    if not isinstance(payload, list):
        return QuotaReading(ok=False, reason="payload is not a list")

    entry = None
    for item in payload:
        if isinstance(item, dict) and item.get("providerId") == "claude":
            entry = item
            break
    if entry is None:
        return QuotaReading(ok=False, reason="no claude provider in payload")

    age: Optional[float] = None
    fetched = entry.get("fetchedAt")
    if isinstance(fetched, str):
        try:
            # Format: 2026-09-15T15:02:49.083Z
            from datetime import datetime, timezone

            ts = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = now - ts.timestamp()
        except Exception:
            age = None

    pcts: dict[str, float] = {}
    resets_at: Optional[str] = None
    for line in entry.get("lines") or []:
        if not isinstance(line, dict) or line.get("type") != "progress":
            continue
        label = str(line.get("label") or "").strip().lower()
        used = line.get("used")
        limit = line.get("limit") or 100
        if not isinstance(used, (int, float)):
            continue
        try:
            pct = (float(used) / float(limit)) * 100.0
        except (TypeError, ZeroDivisionError):
            continue
        pcts[label] = pct
        if label == "session" and isinstance(line.get("resetsAt"), str):
            resets_at = line["resetsAt"]

    if not pcts:
        return QuotaReading(ok=False, reason="no progress lines in payload", age_seconds=age)

    return QuotaReading(
        ok=True,
        window_pct=pcts.get("session"),
        weekly_pct=pcts.get("weekly"),
        weekly_fable_pct=pcts.get("fable"),
        resets_at=resets_at,
        age_seconds=age,
    )


class QuotaGuard:
    """Reads plan usage and decides whether new work may be claimed.

    The reading is cached for ``cache_seconds`` so a 60-second dispatcher tick
    does not hammer the endpoint, and because the upstream value itself only
    changes on the app's own refresh schedule.
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_QUOTA_URL,
        max_age_seconds: float = DEFAULT_QUOTA_MAX_AGE_SECONDS,
        cache_seconds: float = 60.0,
        timeout: float = 5.0,
        fetch_fn=None,
    ) -> None:
        self.url = url
        self.max_age_seconds = max_age_seconds
        self.cache_seconds = cache_seconds
        self.timeout = timeout
        self._fetch_fn = fetch_fn or self._http_fetch
        self._cached: Optional[QuotaReading] = None
        self._cached_at: float = 0.0

    def _http_fetch(self) -> object:
        req = urllib.request.Request(self.url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def read(self, *, now: Optional[float] = None) -> QuotaReading:
        now = time.time() if now is None else now
        if self._cached is not None and (now - self._cached_at) < self.cache_seconds:
            return self._cached

        try:
            payload = self._fetch_fn()
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            # A quota source that is down must not take the dispatcher with it.
            reading = QuotaReading(ok=False, reason=f"fetch failed: {type(exc).__name__}")
        else:
            reading = _parse_openusage(payload, now=now)
            if (
                reading.ok
                and reading.age_seconds is not None
                and reading.age_seconds > self.max_age_seconds
            ):
                reading = QuotaReading(
                    ok=False,
                    window_pct=reading.window_pct,
                    weekly_pct=reading.weekly_pct,
                    weekly_fable_pct=reading.weekly_fable_pct,
                    age_seconds=reading.age_seconds,
                    reason=f"reading is stale ({reading.age_seconds:.0f}s old)",
                )

        self._cached = reading
        self._cached_at = now
        return reading

    def evaluate(
        self, *, threshold_pct: float, allow_on_unknown: bool, now: Optional[float] = None
    ) -> tuple[bool, str]:
        """Return ``(may_claim, human_reason)``.

        ``allow_on_unknown`` is the operator's call about what an unreadable
        quota source means. Defaulting it to True keeps the factory running
        when the menu-bar app is simply not open, which is the common case;
        setting it False makes the guard fail closed.
        """
        reading = self.read(now=now)
        if not reading.ok:
            if allow_on_unknown:
                return True, f"quota unknown ({reading.reason}); proceeding"
            return False, f"quota unknown ({reading.reason}); holding"

        worst = reading.worst_pct
        if worst is None:
            return allow_on_unknown, "quota reading had no usable percentages"
        if worst >= threshold_pct:
            return False, (
                f"quota {worst:.0f}% >= {threshold_pct:.0f}% "
                f"(session={reading.window_pct}, weekly={reading.weekly_pct}, "
                f"fable={reading.weekly_fable_pct})"
            )
        return True, f"quota {worst:.0f}% < {threshold_pct:.0f}%"


@dataclass
class BoardPlan:
    """How many workers a single board may start on this tick."""

    slug: str
    allowed: int
    running: int
    reason: str = ""


@dataclass
class DispatchPlan:
    """The whole tick's decision, in board order."""

    boards: list[BoardPlan] = field(default_factory=list)
    global_running: int = 0
    global_cap: Optional[int] = None
    paused: bool = False
    pause_reason: str = ""

    def allowed_for(self, slug: str) -> int:
        for b in self.boards:
            if b.slug == slug:
                return b.allowed
        return 0


def load_policy_config(cfg: Optional[dict]) -> dict:
    """Extract the policy knobs from a loaded config dict, with defaults.

    Unknown or malformed values fall back to the default rather than raising:
    a typo in config.yaml should not stop the factory, it should log and use
    something safe.
    """
    kan = {}
    if isinstance(cfg, dict):
        kan = cfg.get("kanban") or {}
        if not isinstance(kan, dict):
            kan = {}

    def _pos_int(value, default):
        try:
            ival = int(value)
        except (TypeError, ValueError):
            return default
        return ival if ival >= 1 else default

    def _pct(value, default):
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return default
        return fval if 0 < fval <= 100 else default

    priority = kan.get("board_priority")
    if not isinstance(priority, list) or not all(isinstance(x, str) for x in priority):
        priority = []

    quota = kan.get("quota")
    if not isinstance(quota, dict):
        quota = {}

    runtime = kan.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    soft = runtime.get("soft_seconds")
    if not isinstance(soft, dict):
        soft = {}
    soft_seconds = dict(DEFAULT_RUNTIME_SOFT_SECONDS)
    for role, val in soft.items():
        soft_seconds[str(role)] = _pos_int(val, DEFAULT_RUNTIME_SOFT_SECONDS.get(str(role), 7200))

    return {
        "global_max_in_progress": _pos_int(
            kan.get("global_max_in_progress"), DEFAULT_GLOBAL_MAX_IN_PROGRESS
        ),
        "board_priority": priority,
        "quota_enabled": bool(quota.get("enabled", True)),
        "quota_url": str(quota.get("url") or DEFAULT_QUOTA_URL),
        "quota_threshold_pct": _pct(
            quota.get("max_unattended_pct"), DEFAULT_MAX_UNATTENDED_PCT
        ),
        "quota_max_age_seconds": _pos_int(
            quota.get("max_age_seconds"), DEFAULT_QUOTA_MAX_AGE_SECONDS
        ),
        "quota_allow_on_unknown": bool(quota.get("allow_on_unknown", True)),
        "runtime_soft_seconds": soft_seconds,
        "runtime_hard_multiplier": _pos_int(
            runtime.get("hard_multiplier"), DEFAULT_RUNTIME_HARD_MULTIPLIER
        ),
    }


def order_boards(slugs: list[str], priority: list[str]) -> list[str]:
    """Order boards by the configured priority, then alphabetically.

    Boards absent from ``priority`` sort after the named ones rather than
    being dropped: a newly created board must still get served, just last.
    """
    rank = {slug: i for i, slug in enumerate(priority)}
    return sorted(slugs, key=lambda s: (rank.get(s, len(rank)), s))


def count_running(conn: sqlite3.Connection) -> int:
    """Count tasks in ``running`` on one board's connection."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def plan_dispatch(
    *,
    board_running: dict[str, int],
    policy: dict,
    quota_guard: Optional[QuotaGuard] = None,
    now: Optional[float] = None,
) -> DispatchPlan:
    """Decide per-board spawn budgets for one tick.

    ``board_running`` maps board slug to its current running-task count. The
    caller collects those (cheap COUNT per board) and this function applies
    the global cap, the board order, and the quota guard.

    Running tasks are never touched: the quota guard stops new *claims* only,
    so work already in flight always gets to finish. That is the whole point
    of pausing at a threshold below 100 rather than at the wall.
    """
    cap = policy.get("global_max_in_progress") or DEFAULT_GLOBAL_MAX_IN_PROGRESS
    total_running = sum(int(v) for v in board_running.values())
    plan = DispatchPlan(global_running=total_running, global_cap=cap)

    ordered = order_boards(list(board_running.keys()), policy.get("board_priority") or [])

    if quota_guard is not None and policy.get("quota_enabled", True):
        may_claim, reason = quota_guard.evaluate(
            threshold_pct=policy.get("quota_threshold_pct", DEFAULT_MAX_UNATTENDED_PCT),
            allow_on_unknown=policy.get("quota_allow_on_unknown", True),
            now=now,
        )
        if not may_claim:
            plan.paused = True
            plan.pause_reason = reason
            plan.boards = [
                BoardPlan(slug=s, allowed=0, running=int(board_running[s]), reason=reason)
                for s in ordered
            ]
            return plan

    budget = max(0, cap - total_running)
    for slug in ordered:
        running = int(board_running[slug])
        if budget <= 0:
            plan.boards.append(
                BoardPlan(
                    slug=slug,
                    allowed=0,
                    running=running,
                    reason=f"global cap {cap} reached ({total_running} running)",
                )
            )
            continue
        # The board itself may spawn at most the remaining global budget; the
        # per-board and per-profile caps inside dispatch_once narrow it further.
        plan.boards.append(
            BoardPlan(slug=slug, allowed=budget, running=running, reason="within global cap")
        )
        # Budget is not decremented per board here: dispatch_once reports what
        # it actually spawned, and the caller re-plans on the next tick. Handing
        # each board the full remaining budget in priority order means a higher
        # board always gets first refusal on every free slot.

    return plan


def runtime_limits_for_role(role: str, policy: dict) -> tuple[int, int]:
    """Return ``(soft_seconds, hard_seconds)`` for an assignee role.

    The role is matched on the profile-name suffix (``tb-coder`` -> ``coder``)
    so the same table serves every project's prefix.
    """
    soft_table = policy.get("runtime_soft_seconds") or DEFAULT_RUNTIME_SOFT_SECONDS
    suffix = str(role or "").rsplit("-", 1)[-1].strip().lower()
    soft = soft_table.get(suffix)
    if soft is None:
        soft = max(soft_table.values()) if soft_table else 7200
    mult = policy.get("runtime_hard_multiplier") or DEFAULT_RUNTIME_HARD_MULTIPLIER
    return int(soft), int(soft) * int(mult)
