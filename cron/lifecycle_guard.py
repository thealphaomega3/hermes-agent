"""Gateway lifecycle guard for cron job creation (#30719).

An agent running inside a gateway can schedule a cron job that calls
``hermes gateway restart`` (or ``launchctl kickstart ai.hermes.gateway``
or ``systemctl restart hermes-gateway``).  When the cron fires, the
gateway dies, the supervisor (launchd KeepAlive / systemd Restart=)
revives it, auto-resume picks up the offending session, and the resumed
turn re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command.  It is enforced at
``cron.jobs.create_job`` so it fires on every job-creation path: the
``hermes cron create`` CLI subcommand AND the agent's ``cronjob`` model
tool (which calls ``create_job`` directly, bypassing the CLI layer).

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``hermes gateway``, ``launchctl ... hermes-gateway``,
``systemctl ... hermes-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.  A cron ``prompt`` is fed to a future LLM, not a
shell, so an over-broad substring match on English ("Kong API gateway
autoscaling and restart behavior") would produce a high false-positive
rate without preventing the actual foot-gun, which requires a real
command shape.

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` blocks direct
commands and shell scripts they reference when ``_HERMES_GATEWAY=1``. It also
rejects ``launchctl submit`` in gateway sessions because launchd treats that
primitive as a persistent KeepAlive job, not a one-shot task. ``hermes gateway
stop|restart`` separately refuse to self-target from inside the gateway.
Blocking cron specs at creation time as well means the agent gets an immediate,
informative rejection instead of scheduling a job that will only fail
(silently) when it fires.
"""

from __future__ import annotations

import os
import re
import shlex
import stat
from pathlib import Path
from typing import Callable, Iterator, Optional


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each branch
# is anchored on a concrete command identifier so a match can only fire on
# actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `hermes gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside a
    # gateway is benign (a no-op or "already running" error), and a
    # legitimate cron job might start a sibling profile's gateway.
    r"(?:hermes\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a hermes-gateway label. macOS launchd
    # labels look like `ai.hermes.gateway` / `hermes-gateway`. Requiring the
    # gateway identifier prevents blocking unrelated hermes services (e.g.
    # `launchctl unload ai.hermes.update-checker.plist`).
    # `submit` and `bootstrap` are included alongside the direct verbs
    # (kickstart/etc.): `launchctl submit -l ai.hermes.gateway-<suffix> --
    # <helper-script>` (or `launchctl bootstrap gui/<uid> <plist>`) creates
    # a NEW keepalive job wrapping an arbitrary helper, which is how a
    # blocked direct restart/kill gets laundered into a persistent restart
    # loop instead (#62891) — same foot-gun, indirect shape. Neutral-label
    # submissions that dodge this text anchor are caught separately by
    # `contains_launchctl_submit_command` (execution-aware, label-independent).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    r"|(?:p?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


# A backslash immediately followed by a newline is a POSIX shell line
# continuation — the shell joins the two lines before parsing. Every branch
# above uses `[^\n]*` between its verb and the gateway identifier so the
# match can't span unrelated lines of a longer cron prompt/script, but that
# also means a real multi-line shell invocation split across continuation
# lines (e.g. `launchctl submit \` / `  -l ai.hermes.gateway-... \` / `  -- ...`,
# the exact reported shape in #62891) would otherwise slip past. Collapse
# continuations to a single space before matching, mirroring what the shell
# itself does, rather than loosening `[^\n]*` and risking false positives
# across genuinely separate lines.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    # Continuations are collapsed FIRST: a `#` comment ends at the real newline,
    # and a backslash at the end of a comment line does not continue it, so
    # stripping comments before this substitution could splice a comment's tail
    # onto the next line.
    normalized = _strip_comment_lines(normalized)
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(normalized))


_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")




_ReadRemoteScriptFn = Callable[[str], Optional[str]]


# A heredoc operator (`<< EOF`, `<<-'EOF'`, `<< "EOF"`) introduces a body that
# the shell passes to the command as DATA on stdin, never parses as shell code.
# The delimiter may be quoted; the quoting only controls parameter expansion
# inside the body, not where the body ends.
_HEREDOC_OPENER = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _iter_shell_statement_lines(text: str) -> Iterator[str]:
    """Split *text* into shell statements at NEWLINES THAT ARE REAL SEPARATORS.

    A bare ``splitlines()`` is wrong twice over, and both mistakes fail the
    same way: they hand fragments of non-shell text to the lexer, which
    tokenizes them into paths the caller then treats as referenced scripts.

    1. A newline inside a quoted string is data, not a separator. Splitting a
       multi-line ``python3 -c "...."`` on it lexes each line of PYTHON as its
       own shell command, so ``with open('/some/data.jsonl') as f:`` yields
       ``/some/data.jsonl`` as an "executable" - and a data file that is a
       directory or over the size cap then fails closed as unsafe, blocking an
       innocent command outright.
    2. A heredoc body is stdin data for the same reason, so ``python3 - <<'EOF'``
       followed by Python source hits the identical failure.

    Tracking quotes and skipping heredoc bodies keeps the guard scanning only
    text the shell would actually execute. The fail-closed behaviour for
    genuinely unreadable or oversized SCRIPTS is deliberate and unchanged - the
    bug was never that it fails closed, it is that these inputs were not
    scripts in the first place.
    """
    buffer: list[str] = []
    quote: Optional[str] = None
    escaped = False
    pending_heredocs: list[str] = []
    heredoc_delimiter: Optional[str] = None

    def _flush() -> Iterator[str]:
        nonlocal buffer, pending_heredocs, heredoc_delimiter
        line = "".join(buffer)
        buffer = []
        if heredoc_delimiter is None:
            for match in _HEREDOC_OPENER.finditer(line):
                pending_heredocs.append(match.group(2))
            yield line
        if pending_heredocs and heredoc_delimiter is None:
            heredoc_delimiter = pending_heredocs.pop(0)

    for char in text:
        if heredoc_delimiter is not None:
            # Inside a heredoc body: consume verbatim until the delimiter line.
            if char == "\n":
                if "".join(buffer).strip() == heredoc_delimiter:
                    heredoc_delimiter = pending_heredocs.pop(0) if pending_heredocs else None
                buffer = []
            else:
                buffer.append(char)
            continue

        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\" and quote != "'":
            buffer.append(char)
            escaped = True
            continue
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            buffer.append(char)
            continue
        if char == "\n":
            yield from _flush()
            continue
        buffer.append(char)

    if heredoc_delimiter is None:
        yield from _flush()


_COMMENT_LINE = re.compile(r"(?m)^[ \t]*#.*$")


def _strip_comment_lines(text: str) -> str:
    """Blank out whole-line ``#`` comments before a raw-regex lifecycle scan.

    ``_iter_command_segments`` already drops comments (``lexer.commenters``),
    but ``contains_gateway_lifecycle_command`` is a raw regex over the whole
    text and has no such notion. A script that merely DOCUMENTS the foot-gun -
    ``# never run `hermes gateway restart` from inside a card`` - therefore
    reads as the foot-gun itself and blocks the script that contains it.

    Lines are blanked rather than deleted so line structure is preserved: every
    lifecycle branch uses ``[^\\n]*`` between its verb and the gateway
    identifier specifically so a match cannot span unrelated lines, and
    collapsing lines here would hand it exactly that opportunity.
    """
    return _COMMENT_LINE.sub("", text)


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments, honoring quotes and comments."""
    normalized = command.replace("\\\n", "")
    for line in _iter_shell_statement_lines(normalized):
        try:
            lexer = shlex.shlex(
                line,
                posix=True,
                punctuation_chars=";&|()",
            )
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            continue

        segment: list[str] = []
        for token in tokens:
            if token and set(token) <= _CONTROL_CHARS:
                if segment:
                    yield segment
                    segment = []
                continue
            segment.append(token)
        if segment:
            yield segment


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    for index, token in enumerate(segment):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        return index
    return None


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: the label of a submitted/bootstrapped job is
    chosen by whoever writes it, so a neutral name (``ai.hermes.svc-reload-tmp``)
    defeats any label-anchored regex (#62891, second reproduction). Both verbs
    register a NEW persistent launchd job (``submit`` jobs get KeepAlive
    semantics; ``bootstrap`` loads an arbitrary plist), which is never safe to
    do from inside the gateway process.
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        if Path(segment[index]).name == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _resolve_terminal_script_path(candidate: str, cwd: Optional[str]) -> Optional[Path]:
    """Resolve a candidate script reference, or None when it cannot be a path.

    Returns None instead of raising. Every input here is UNTRUSTED text that
    was tokenized out of a command - when the recursion walks a binary, the
    tokenizer emits machine-code fragments, and a fragment starting with ``~``
    reaches ``expanduser``. That raises ``RuntimeError`` (NOT OSError/ValueError)
    when the user database has no entry for the "username" it parsed out, and
    the exception propagated all the way through ``terminal_tool`` and killed
    the whole tool call - 23 times across two profiles in one day.

    ``Path()`` and ``Path.cwd()`` are inside the guard for the same reason: a
    NUL byte raises ``ValueError`` from the constructor, and ``Path.cwd()``
    raises ``OSError`` when the process cwd has been unlinked. A guard must
    never be the thing that crashes the operation it guards, so the failure
    mode here is "this token is not a resolvable script path" - which is the
    truth - rather than an exception.
    """
    try:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        return path
    except (RuntimeError, ValueError, OSError):
        return None


def _iter_referenced_shell_scripts(
    command: str,
    *,
    cwd: Optional[str] = None,
) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell.

    Unresolvable candidates are dropped rather than yielded: the resolver
    returns None for tokens that cannot be a path at all, and a caller that
    treated None as a script would reintroduce the crash one frame later.
    """

    def _emit(candidate: str) -> Iterator[Path]:
        resolved = _resolve_terminal_script_path(candidate, cwd)
        if resolved is not None:
            yield resolved

    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        executable = segment[index]
        try:
            executable_name = Path(executable).name
        except ValueError:
            # Embedded NUL from tokenized machine code - not an executable.
            continue

        if executable_name in {".", "source"}:
            if len(segment) > index + 1:
                yield from _emit(segment[index + 1])
            continue

        if executable_name in _SHELL_EXECUTABLES:
            arguments = segment[index + 1 :]
            arg_index = 0
            while arg_index < len(arguments):
                argument = arguments[arg_index]
                if argument == "--":
                    arg_index += 1
                    break
                if argument in {"-c", "--command"}:
                    break
                if argument in _SHELL_OPTIONS_WITH_VALUES:
                    arg_index += 2
                    continue
                if argument.startswith("-"):
                    arg_index += 1
                    continue
                break
            if arg_index < len(arguments) and arguments[arg_index] not in {
                "-c",
                "--command",
            }:
                yield from _emit(arguments[arg_index])
            continue

        # A bare "/" token is pathlib's division operator in Python sources
        # (e.g. `Path.home() / ".hermes"`), not an executable reference.
        # Resolving it walks to the filesystem root and fails the
        # regular-file check below, hard-blocking innocent .py scripts
        # (#77131). Skip pure-separator tokens.
        if executable.strip("/"):
            if "/" in executable or executable.endswith((".sh", ".bash", ".zsh")):
                yield from _emit(executable)


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` for recursive scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None or Path(segment[index]).name not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in {"-c", "--command"}:
                yield arguments[arg_index + 1]
                break


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


def _read_referenced_script(path: Path) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        # ValueError: os.open raises it, NOT OSError, for a path containing an
        # embedded NUL byte - which is exactly what the recursion feeds in when
        # a command names a real binary (Godot.app/.../Godot): the binary's
        # machine code is tokenized and junk paths come back out. The caller at
        # _contains_unsafe_gateway_action already catches ValueError from
        # Path.resolve for this same reason (#76762), but the fix stopped one
        # call short, so the guard still crashed here and took the whole
        # terminal tool down with it. A guarded path must never crash the guard.
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None, True
        # Read a bounded chunk first — even for oversized files, the first
        # chunk tells us if this is a binary (NUL bytes) that should be
        # skipped as "nothing to scan" rather than failing closed (#76762).
        data = os.read(descriptor, _MAX_REFERENCED_SCRIPT_BYTES + 1)
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    # A NUL byte in the first chunk means this is a binary (ELF/Mach-O/
    # PE), not a shell script — scanning its decoded contents would
    # tokenize machine code and feed junk paths into the recursion
    # (including a `ValueError: embedded null byte` from Path.resolve,
    # #76762). Treat it as "nothing to scan" rather than unsafe: a binary
    # executed by the user is not a referenced *shell script*.
    if b"\x00" in data:
        return None, False
    if len(data) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return data.decode("utf-8", errors="replace"), False


def _contains_unsafe_gateway_action(
    command: str,
    *,
    cwd: Optional[str],
    depth: int,
    visited: set[Path],
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    if contains_gateway_lifecycle_command(command) or contains_launchctl_submit_command(
        command
    ):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    for payload in _iter_shell_command_payloads(command):
        if _contains_unsafe_gateway_action(
            payload,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True

    for script_path in _iter_referenced_shell_scripts(command, cwd=cwd):
        try:
            resolved = script_path.resolve(strict=False)
        except (OSError, ValueError):
            # OSError: unreadable/long paths. ValueError: embedded NUL byte
            # from a binary's decoded contents tokenized as a path — a
            # guarded path must never crash the guard (#76762).
            resolved = script_path
        if resolved in visited:
            continue
        visited.add(resolved)
        script_text, unsafe = _read_referenced_script(script_path)
        if unsafe:
            return True
        if script_text is None and read_remote_script is not None:
            # Local path missing; try the remote backend if one is available.
            script_text = read_remote_script(str(script_path))
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's
        # directory, not the original command's cwd.
        script_dir = _resolve_script_directory(str(resolved)) or cwd
        if script_text and _contains_unsafe_gateway_action(
            script_text,
            cwd=script_dir,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str,
    *,
    cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts."""
    return _contains_unsafe_gateway_action(
        command,
        cwd=cwd,
        depth=0,
        visited=set(),
        read_remote_script=read_remote_script,
    )




def _resolve_script_path(script_path: str) -> Path:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.
    """
    from hermes_constants import get_hermes_home

    try:
        raw = Path(script_path).expanduser()
    except (RuntimeError, ValueError, OSError):
        # `~nosuchuser/...` raises RuntimeError here just as it does on the
        # terminal path. A cron `script` value is user-supplied rather than
        # tokenizer output, so this is far less likely - but "less likely" is
        # not a reason for the guard to be the thing that crashes job
        # creation. Fall back to the literal, unexpanded path: it will simply
        # fail to open, which `_read_script_for_scanning` already handles by
        # returning empty text so ordinary scheduler path validation reports
        # the bad path with a useful message.
        raw = Path(script_path.lstrip("~") or ".")
    if raw.is_absolute():
        return raw
    return get_hermes_home() / "scripts" / raw


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded terminal-script scanner.

    Non-regular or oversized inputs fail closed by returning a lifecycle-shaped
    sentinel, while missing/unreadable paths remain empty so ordinary scheduler
    path validation can report them.
    """
    script_text, unsafe = _read_referenced_script(_resolve_script_path(script_path))
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a
    gateway-lifecycle command pattern.

    ``prompt`` is scanned directly.  ``script``, when supplied, is read from
    disk and concatenated for the scan.  Both are considered together so a
    job cannot slip through by splitting the command across the prompt and
    the script.

    Callers should let the exception propagate when they want the create to
    fail with a ``ValueError``-shaped error (the agent's ``cronjob`` tool
    surfaces this as a tool error; the CLI prints it in red and exits 1).
    """
    combined = prompt or ""
    python_script = False
    if script:
        python_script = _resolve_script_path(script).suffix == ".py"
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if python_script:
        # Python is executed by the interpreter, never through a POSIX
        # shell: the shell-script reference walk is a false-positive
        # generator on Python sources (pathlib's "/" operator resolves to
        # the filesystem root and trips the regular-file check, blocking
        # every innocent .py cron script, #77131). The direct command
        # regex below still scans the full text, so a literal
        # `hermes gateway restart` embedded in a .py script is still
        # blocked. Non-regular/oversized script files still fail closed
        # via the lifecycle-shaped sentinel in _read_script_for_scanning.
        unsafe = contains_gateway_lifecycle_command(combined)
    else:
        script_dir = _resolve_script_directory(script) if script else None
        unsafe = contains_gateway_lifecycle_command_or_referenced_script(
            combined,
            cwd=script_dir,
        )
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
