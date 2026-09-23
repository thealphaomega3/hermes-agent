"""Regression tests for iteration-limit exit normalization (#61631)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import _record_kanban_budget_exhausted, finalize_turn
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tests.hermes_cli.test_kanban_db import (
    kanban_home,  # noqa: F401 — shared temp-HERMES_HOME + kanban DB fixture
)


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
    )
















@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once_with(
        conn,
        "task-123",
        error=(
            "Iteration budget exhausted (60/60) — task could not complete "
            "within the allowed iterations"
        ),
        outcome="timed_out",
        force_trip=True,
        release_claim=True,
        end_run=True,
        event_payload_extra={"budget_used": 60, "budget_max": 60},
    )


def test_budget_exhaustion_trips_breaker_on_first_occurrence(kanban_home):
    """Real-path contract: a run that spent its whole iteration budget blocks the
    card for an operator on the FIRST occurrence instead of being respawned once.

    The respawn is exactly what burned a second full budget — the retry started
    10s later, found the prior run's work still uncommitted in the worktree, and
    spent another 300 calls. A same-budget rerun cannot fit what the first run
    could not fit, so ``_record_kanban_budget_exhausted`` passes
    ``force_trip=True`` and the ``gave_up`` event is stamped ``sticky``, which
    makes ``recompute_ready`` hold the card.
    """
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="budget burner")
        assert kb.claim_task(conn, tid) is not None
        running = kb.get_task(conn, tid)
        assert running is not None
        assert running.status == "running"
        assert running.current_run_id is not None

    _record_kanban_budget_exhausted(tid, 300, 300, logging.getLogger(__name__))

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.current_run_id is None
        assert task.claim_lock is None

        gave_up = [event for event in kb.list_events(conn, tid) if event.kind == "gave_up"]
        assert len(gave_up) == 1
        payload = gave_up[0].payload or {}
        assert payload["sticky"] is True
        assert payload["trigger_outcome"] == "timed_out"
        assert payload["budget_used"] == 300
        assert payload["budget_max"] == 300
        assert "Iteration budget exhausted (300/300)" in payload["error"]

        runs = kb.list_runs(conn, tid)
        assert [run.outcome for run in runs] == ["gave_up"]
        ended_at = runs[0].ended_at
        assert ended_at is not None

        # Dispatcher tick: the sticky block is held for an operator, not
        # promoted back to ``ready`` for a same-budget respawn.
        assert kb.recompute_ready(conn) == 0
        still_blocked = kb.get_task(conn, tid)
        assert still_blocked is not None
        assert still_blocked.status == "blocked"

        failures_before = task.consecutive_failures
        error_before = task.last_failure_error

    # A second exit path re-recording the same exhaustion is a no-op on every
    # piece of state the dispatcher reads: no new run, no reopened card.
    _record_kanban_budget_exhausted(tid, 300, 300, logging.getLogger(__name__))

    with kbc.connect_closing() as conn:
        again = kb.get_task(conn, tid)
        assert again is not None
        assert again.status == "blocked"
        assert again.current_run_id is None
        assert again.consecutive_failures == failures_before
        assert again.last_failure_error == error_before
        assert [(run.outcome, run.ended_at) for run in kb.list_runs(conn, tid)] == [
            ("gave_up", ended_at)
        ]


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


def test_bounded_fallback_records_kanban_failure_when_interrupted(monkeypatch):
    """When budget is exhausted and the turn was interrupted,
    ``finalize_turn`` must still record a terminal kanban failure via
    the bounded fallback path (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-456")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    # Budget exhausted (60/60), interrupted, no fallback-eligible exit_reason
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )

    # The bounded fallback must fire even though interrupted=True
    # makes budget_fallback_eligible=False.
    record.assert_called_once()
    args, kwargs = record.call_args
    assert args[1] == "task-456"
    assert kwargs["outcome"] == "timed_out"
    assert kwargs["force_trip"] is True
    assert kwargs["release_claim"] is True
    assert kwargs["end_run"] is True
    assert kwargs["event_payload_extra"]["budget_used"] == 60
    assert kwargs["event_payload_extra"]["budget_max"] == 60


def test_bounded_fallback_records_kanban_failure_when_failed(monkeypatch):
    """When budget is exhausted and the turn failed,
    the bounded fallback must still record a terminal kanban failure (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-789")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=True,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="provider_failure",
    )

    record.assert_called_once()
    args, kwargs = record.call_args
    assert args[1] == "task-789"
    assert kwargs["outcome"] == "timed_out"
    assert kwargs["force_trip"] is True


def test_bounded_fallback_does_not_fire_without_kanban_task(monkeypatch):
    """When budget is exhausted and interrupted but no kanban task is
    active, the bounded fallback must NOT fire (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )

    record.assert_not_called()


def test_bounded_fallback_does_not_fire_when_budget_not_exhausted(monkeypatch):
    """When budget is NOT exhausted but turn is interrupted and a kanban
    task is active, the bounded fallback must NOT fire (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-999")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent(budget_remaining=60)

    # api_call_count=10, max_iterations=60 — budget NOT exhausted
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=10,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )

    record.assert_not_called()


@pytest.mark.parametrize("scope", ["child", "non-owner"])
def test_budget_exhausted_child_does_not_record_parent_kanban_timeout(monkeypatch, scope):
    """An in-process delegate_task child (or cron run) inherits ``HERMES_KANBAN_TASK`` from
    the dispatcher worker; exhausting ITS budget must not record ``timed_out`` against the
    parent's task or release the parent's claim (#112817)."""
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    ctx = delegated_child_context if scope == "child" else non_dispatcher_owned_context
    with ctx():
        finalize_turn(
            agent,
            final_response=None,
            api_call_count=60,
            interrupted=True,
            failed=False,
            messages=[{"role": "user", "content": "task"}],
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn",
            user_message="task",
            original_user_message="task",
            _should_review_memory=False,
            _turn_exit_reason="interrupted_by_user",
        )

    record.assert_not_called()


def test_finalize_turn_starts_the_title_upgrade_the_prologue_held_back():
    """#117296: the turn prologue leaves a same-endpoint title upgrade unstarted on the agent; the finalizer
    is the only place that may start it, and only once the model request is done."""
    import threading

    ran = threading.Event()
    agent = _LimitAgent()
    agent._deferred_title_upgrade = threading.Thread(target=ran.set, daemon=True)
    _finalize(agent, final_response="done", exit_reason="text_response(1)", api_call_count=1)
    assert ran.wait(timeout=5), "deferred title upgrade never started"
    assert agent._deferred_title_upgrade is None
