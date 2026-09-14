"""Parse vLLM Buildkite job logs into structured failure signatures.

Vendored from the vllm-pytorch-ci-triage package (parse_log/strip_markers + the
dataclasses they need). Strips ANSI escape codes and BKT timestamp markers, then
extracts per-test signatures: test id, exception class/message, and the raw section
body from pytest's FAILURES block. Extraction is position-independent -- a failure
far from the end of a huge log is still captured.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailedTest:
    """A single failed test with its exception signature."""

    test_id: str
    pytest_exception_class: str = (
        ""  # The exception class pytest named on its inline FAILED/ERROR summary line.
    )
    exception_chain: str = ""  # This test's own FAILURES section body
    inline_message: str = ""
    test_is_infra: bool = False


@dataclass
class PytestResult:
    """One pytest session's output within a Buildkite job log."""

    test_failures: list[FailedTest] = field(default_factory=list)
    pytest_summary: str = ""
    expected_test_failure_count: int | None = None


@dataclass
class ParsedLog:
    """Parsed log output from a Buildkite job.

    Either pytest_results is populated (pytest failures) or error_excerpt holds the
    whole cleaned log (a build/crash before pytest ran).
    """

    pytest_results: list[PytestResult] = field(default_factory=list)
    error_excerpt: str = ""
    job_is_infra: bool = False


@dataclass(frozen=True)
class FailureContextConfig:
    """Bounds and context sizes used by :func:`extract_failure_context`."""

    max_failure_window_instances_per_type: int = 3
    raw_tail_line_count: int = 400
    failure_window_context_before_lines: int = 12
    failure_window_context_after_lines: int = 80

    def __post_init__(self) -> None:
        if self.max_failure_window_instances_per_type < 1:
            raise ValueError(
                "max_failure_window_instances_per_type must be at least one"
            )
        if self.raw_tail_line_count < 0:
            raise ValueError("raw_tail_line_count must not be negative")
        if self.failure_window_context_before_lines < 0:
            raise ValueError("failure_window_context_before_lines must not be negative")
        if self.failure_window_context_after_lines < 0:
            raise ValueError("failure_window_context_after_lines must not be negative")


TIMESTAMP_RE = re.compile(r"^\[[\d\-T:Z]+\]\s*")
FAILED_TEST_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+/\S*\.py\S*)")
PYTEST_SUMMARY_RE = re.compile(
    r"=+\s+.*\d+\s+(?:failed|error|passed|skipped|warning|deselected).*\bin\s+\d.*=+"
)
SUMMARY_FAILED_COUNT_RE = re.compile(r"(\d+)\s+failed")
SUMMARY_ERROR_COUNT_RE = re.compile(r"(\d+)\s+error")
TEST_SECTION_HEADER_RE = re.compile(r"_{1,}\s+(.+?)\s+_{1,}")

# The inline path trusts pytest: the text after " - " on a FAILED/ERROR line is
# ExceptionInfo.exconly() output -- "{ExcClass}: {message}". The message is
# optional: exconly() emits a bare class name when the exception has no message.
FAILED_INLINE_EXC_RE = re.compile(
    r"(?:FAILED|ERROR)\s+(.+?)\s+-\s+"
    r"([A-Za-z_][\w.]*(?:::[\w.]+)*)"
    r"(?::\s*(.*))?"
)
FAILED_INLINE_ASSERT_RE = re.compile(r"(?:FAILED|ERROR)\s+(.+?)\s+-\s+(assert\s+.*)")

INFRA_PATTERNS = [
    re.compile(r"nvidia-container-cli", re.IGNORECASE),
    re.compile(r"CUDA driver initialization failed", re.IGNORECASE),
    re.compile(r"exit status 137"),
    re.compile(r"exit(?:ed with)? status 125", re.IGNORECASE),
    re.compile(r"Free memory on device cuda:\d+.*less than desired"),
    re.compile(r"docker.*pull", re.IGNORECASE),
    re.compile(r"command hook exited with status", re.IGNORECASE),
    re.compile(r"toomanyrequests", re.IGNORECASE),
    re.compile(r"Data limit exceeded", re.IGNORECASE),
    re.compile(r"Connection refused", re.IGNORECASE),
    re.compile(r"no space left on device", re.IGNORECASE),
    re.compile(r"manifest unknown", re.IGNORECASE),
    re.compile(r"not found: manifest", re.IGNORECASE),
]


FAILURE_WINDOW_TYPES = (
    "engine_core_failure",
    "worker_failure",
    "api_server_failure",
    "python_traceback",
    "signal_or_process_exit",
    "cuda_nccl_or_oom",
    "import_or_linker_failure",
    "command_failure",
)

# These patterns classify anchor lines in precedence order. A line belongs to at
# most one type, so a CUDA exception inside an EngineCore traceback is represented
# by the EngineCore occurrence rather than creating a second anchor on that line.
_FAILURE_ANCHOR_PATTERNS = {
    "engine_core_failure": re.compile(
        r"(?:EngineCore\s+failed\s+to\s+start|"
        r"Engine\s+core\s+initialization\s+failed|"
        r"Process\s+EngineCore\b|"
        r"\(EngineCore\s+pid=\d+\).*"
        r"(?:Traceback|\b(?:ERROR|CRITICAL)\b|\b[A-Za-z_][\w.]*Error:)|"
        r"\bEngineCore\s+pid=\d+.*\b(?:Traceback|[A-Za-z_][\w.]*Error:))",
        re.IGNORECASE,
    ),
    "worker_failure": re.compile(
        r"(?:\b(?:vLLM\s+)?(?:worker|WorkerProc|RayWorkerWrapper|TP\s+worker|DP\s+worker|"
        r"Ray\s+worker)\b.*(?:failed|fatal|error|exception|traceback|exited)|"
        r"(?:worker|rank\s*\d+).*\b(?:fatal|failed|crashed)\b)",
        re.IGNORECASE,
    ),
    "api_server_failure": re.compile(
        r"(?:\bAPIServer\b.*(?:traceback|failed|fatal|error|exception|exited)|"
        r"\b(?:API|HTTP)\s+server\b.*(?:failed|fatal|error|exception|exited)|"
        r"Server\s+exited\s+unexpectedly|"
        r"server\s+subprocess.*(?:failed|exited)|"
        r"(?:api|http)\s+server.*\b(?:died|death)\b)",
        re.IGNORECASE,
    ),
    "signal_or_process_exit": re.compile(
        r"(?:\bSIG(?:ABRT|SEGV|KILL)\b|"
        r"\bsignal\s+\d+\b|"
        r"\bProcessExitedException\b|"
        r"\b(?:child|subprocess|process)\b.*(?:exit(?:ed)?|terminated|killed)"
        r"(?:\s+with)?\s+(?:code|status|signal)?\s*\d+)",
        re.IGNORECASE,
    ),
    "cuda_nccl_or_oom": re.compile(
        r"(?:\bCUDA\b|\bNCCL\b|\b(?:OutOfMemoryError|CUDAOutOfMemoryError)\b|"
        r"out\s+of\s+memory|"
        r"free\s+memory\s+on\s+device\s+cuda:\d+.*less\s+than\s+desired|"
        r"GPU.*(?:memory|OOM)|"
        r"(?:watchdog|peer)\s+failure)",
        re.IGNORECASE,
    ),
    "import_or_linker_failure": re.compile(
        r"(?:\b(?:ImportError|ModuleNotFoundError)\b|"
        r"undefined\s+symbol|"
        r"cannot\s+open\s+shared\s+object\s+file|"
        r"no\s+such\s+file\s+or\s+directory.*\.so)",
        re.IGNORECASE,
    ),
    "command_failure": re.compile(
        r"(?:\b(?:The\s+)?command\s+(?:exited|failed)\s+with\s+(?:status|code)\s+\d+|"
        r"\buser\s+command\s+error\b|"
        r"\bplugin\b.*\b(?:command|hook)\b.*\b(?:exited|failed)\s+with\s+(?:status|code)\s+\d+)",
        re.IGNORECASE,
    ),
    "python_traceback": re.compile(r"Traceback\s+\(most\s+recent\s+call\s+last\):"),
}

_PROCESS_RE = re.compile(
    r"\(([^()\n]*\bpid=\d+)[^()\n]*\)|\b(Process\s+[A-Za-z][\w.-]*(?:\s+pid=\d+)?)\b"
)
_TEST_OR_COMMAND_MARKER_RE = re.compile(
    r"(?:\+\+\+.*(?:Command|pytest)|"
    r"(?:FAILED|ERROR|PASSED)\s+\S+|"
    r"(?:^|\s)(?:pytest|python|uv|docker)\s+\S+)",
    re.IGNORECASE,
)


def get_test_signature(failed_test: "FailedTest") -> tuple[str, str]:
    """Build the diff key for a failing test.

    The exception_chain is deliberately excluded: it is not stable build-to-build
    across the torch-nightly/baseline A/B.

    Args:
        failed_test: Parsed pytest failure to key.

    Returns:
        The (test_id, exception_class) pair.
    """
    return (failed_test.test_id, failed_test.pytest_exception_class)


def strip_markers(text: str) -> str:
    """Remove escape sequences from raw Buildkite log text.

    Filters:
        - ANSI CSI sequences (\\x1b[...): colors, cursor movement, erase-to-EOL
        - BKT timestamp markers (\\x1b_bk;t=<ms>\\x07)
        - OSC sequences (\\x1b]...\\x07): inline images (1338), hyperlinks (1339)
    """
    ansi_regex = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
    osc_regex = re.compile(r"\x1b[\]_][^\x07]*\x07")
    return osc_regex.sub("", ansi_regex.sub("", text))


def clean_failure_context_lines(text: str) -> list[str]:
    """Return display lines with only transport/presentation noise removed."""
    return [TIMESTAMP_RE.sub("", line) for line in strip_markers(text).splitlines()]


_FAILURE_ANCHOR_PRECEDENCE = (
    "engine_core_failure",
    "worker_failure",
    "api_server_failure",
    "signal_or_process_exit",
    "cuda_nccl_or_oom",
    "import_or_linker_failure",
    "command_failure",
    "python_traceback",
)


def _classify_failure_anchor(line: str) -> str | None:
    for window_type in _FAILURE_ANCHOR_PRECEDENCE:
        if _FAILURE_ANCHOR_PATTERNS[window_type].search(line):
            return window_type
    return None


def _nearest_match(
    lines: list[str],
    anchor_index: int,
    start_index: int,
    end_index: int,
    pattern: re.Pattern[str],
) -> str:
    candidates = []
    for index in range(start_index, end_index):
        match = pattern.search(lines[index])
        if match:
            value = next((group for group in match.groups() if group), "")
            candidates.append((abs(index - anchor_index), value.strip()))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else ""


def _failure_window_instance(
    lines: list[str],
    window_type: str,
    anchor_index: int,
    config: FailureContextConfig,
) -> dict[str, Any]:
    start_index = max(0, anchor_index - config.failure_window_context_before_lines)
    end_index = min(
        len(lines), anchor_index + config.failure_window_context_after_lines + 1
    )
    return {
        "window_type": window_type,
        "anchor_line": anchor_index + 1,
        "start_line": start_index + 1,
        "end_line": end_index,
        "process": _nearest_match(
            lines, anchor_index, start_index, end_index, _PROCESS_RE
        ),
        "nearest_marker": _nearest_match(
            lines,
            anchor_index,
            start_index,
            end_index,
            _TEST_OR_COMMAND_MARKER_RE,
        ),
        "text": "\n".join(lines[start_index:end_index]),
    }


def _merge_failure_window_intervals(
    lines: list[str], instances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge selected failure-window intervals in chronological order.

    The instances have already been sampled independently for each window type.
    Merging happens only after that sampling, so the per-type matched/emitted
    counts remain counts of the original occurrences rather than counts of the
    rendered intervals.  A one-line gap is not allowed: directly adjacent
    intervals have ``next.start_line == current.end_line + 1`` and are merged.
    """
    ordered = sorted(
        instances,
        key=lambda instance: (
            instance["start_line"],
            instance["end_line"],
            instance["anchor_line"],
            instance["window_type"],
        ),
    )
    merged: list[dict[str, Any]] = []

    for instance in ordered:
        if not merged or instance["start_line"] > merged[-1]["end_line"] + 1:
            merged.append(
                {
                    "window_type": instance["window_type"],
                    "window_types": [instance["window_type"]],
                    # Keep the singular fields for compatibility with the
                    # Phase 1 shape.  ``anchor_lines`` and ``anchors`` retain
                    # every occurrence when an interval contains more than one.
                    "anchor_line": instance["anchor_line"],
                    "anchor_lines": [instance["anchor_line"]],
                    "start_line": instance["start_line"],
                    "end_line": instance["end_line"],
                    "process": instance["process"],
                    "processes": ([instance["process"]] if instance["process"] else []),
                    "nearest_marker": instance["nearest_marker"],
                    "nearest_markers": (
                        [instance["nearest_marker"]]
                        if instance["nearest_marker"]
                        else []
                    ),
                    "anchors": [
                        {
                            "window_type": instance["window_type"],
                            "anchor_line": instance["anchor_line"],
                            "process": instance["process"],
                            "nearest_marker": instance["nearest_marker"],
                        }
                    ],
                    "text": "\n".join(
                        lines[instance["start_line"] - 1 : instance["end_line"]]
                    ),
                }
            )
            continue

        current = merged[-1]
        current["end_line"] = max(current["end_line"], instance["end_line"])
        if instance["window_type"] not in current["window_types"]:
            current["window_types"].append(instance["window_type"])
        current["anchor_lines"].append(instance["anchor_line"])
        current["anchors"].append(
            {
                "window_type": instance["window_type"],
                "anchor_line": instance["anchor_line"],
                "process": instance["process"],
                "nearest_marker": instance["nearest_marker"],
            }
        )
        if instance["process"]:
            current["processes"].append(instance["process"])
        if instance["nearest_marker"]:
            current["nearest_markers"].append(instance["nearest_marker"])
        current["window_type"] = (
            current["window_types"][0]
            if len(current["window_types"]) == 1
            else "merged"
        )
        current["text"] = "\n".join(
            lines[current["start_line"] - 1 : current["end_line"]]
        )

    return merged


def extract_failure_context(
    text: str,
    config: FailureContextConfig | None = None,
    *,
    max_failure_window_instances_per_type: int | None = None,
    raw_tail_line_count: int | None = None,
    failure_window_context_before_lines: int | None = None,
    failure_window_context_after_lines: int | None = None,
) -> dict[str, Any]:
    """Extract bounded, structured failure context from a complete Buildkite log.

    The complete cleaned log is scanned for classified anchor lines, but only the
    first configured number of windows per type is retained. The selected intervals
    are then sorted and merged when they overlap or are directly adjacent. The
    returned object is JSON-serializable and contains the complete match counts so
    callers can see when sampling truncated a noisy failure, as well as the number
    of post-merge intervals rendered below those counts.
    """
    if config is not None and any(
        value is not None
        for value in (
            max_failure_window_instances_per_type,
            raw_tail_line_count,
            failure_window_context_before_lines,
            failure_window_context_after_lines,
        )
    ):
        raise ValueError("config cannot be combined with individual overrides")
    if config is None:
        config = FailureContextConfig(
            max_failure_window_instances_per_type=(
                max_failure_window_instances_per_type
                if max_failure_window_instances_per_type is not None
                else 3
            ),
            raw_tail_line_count=(
                raw_tail_line_count if raw_tail_line_count is not None else 400
            ),
            failure_window_context_before_lines=(
                failure_window_context_before_lines
                if failure_window_context_before_lines is not None
                else 12
            ),
            failure_window_context_after_lines=(
                failure_window_context_after_lines
                if failure_window_context_after_lines is not None
                else 80
            ),
        )

    lines = clean_failure_context_lines(text)
    anchors: dict[str, list[int]] = {
        window_type: [] for window_type in FAILURE_WINDOW_TYPES
    }
    for line_index, line in enumerate(lines):
        window_type = _classify_failure_anchor(line)
        if window_type is not None:
            anchors[window_type].append(line_index)

    failure_windows = []
    selected_windows: list[dict[str, Any]] = []
    for window_type in FAILURE_WINDOW_TYPES:
        matches = anchors[window_type]
        selected = matches[: config.max_failure_window_instances_per_type]
        instances = [
            _failure_window_instance(lines, window_type, index, config)
            for index in selected
        ]
        selected_windows.extend(instances)
        failure_windows.append(
            {
                "window_type": window_type,
                "matched_instance_count": len(matches),
                "emitted_instance_count": len(instances),
                "instances_truncated": len(matches) > len(instances),
                "instances": instances,
            }
        )

    rendered_windows = _merge_failure_window_intervals(lines, selected_windows)
    for summary in failure_windows:
        summary["rendered_interval_count"] = sum(
            summary["window_type"] in interval["window_types"]
            for interval in rendered_windows
        )
    tail_count = min(config.raw_tail_line_count, len(lines))
    if tail_count:
        tail_start_line = len(lines) - tail_count + 1
        raw_tail = "\n".join(lines[-tail_count:])
    else:
        tail_start_line = 0
        raw_tail = ""
    return {
        "line_count": len(lines),
        "failure_window_context_before_lines": config.failure_window_context_before_lines,
        "failure_window_context_after_lines": config.failure_window_context_after_lines,
        "max_failure_window_instances_per_type": config.max_failure_window_instances_per_type,
        "failure_windows": failure_windows,
        "rendered_interval_count": len(rendered_windows),
        "windows_in_chronological_order": rendered_windows,
        "raw_tail": {
            "start_line": tail_start_line,
            "end_line": len(lines) if tail_count else 0,
            "line_count": tail_count,
            "text": raw_tail,
        },
    }


def parse_log(text: str) -> ParsedLog:
    """Clean a raw log and extract structured failure information.

    Args:
        text: Raw log text from Buildkite.

    Returns:
        Parsed log with extracted failure signatures.
    """
    cleaned = strip_markers(text)
    lines = cleaned.splitlines()

    extraction = _extract_pytest_failures(lines)

    if not extraction.pytest_results:
        return ParsedLog(error_excerpt=cleaned, job_is_infra=_matches_infra(cleaned))

    for pytest_result in extraction.pytest_results:
        for failure in pytest_result.test_failures:
            # Both fields hold only what pytest scoped to this specific test:
            #   - pytest_exception_class: the class pytest named on the inline
            #     FAILED/ERROR summary line (set at construction); empty otherwise.
            #   - exception_chain: this test's own FAILURES section body.
            section_body = _get_section_body(failure.test_id, extraction.section_bodies)
            if section_body is not None:
                failure.exception_chain = section_body
            elif failure.pytest_exception_class:
                failure.exception_chain = (
                    f"{failure.pytest_exception_class}: {failure.inline_message}"
                )
        for failure in pytest_result.test_failures:
            failure.test_is_infra = _matches_infra(failure.exception_chain)

    return ParsedLog(pytest_results=extraction.pytest_results)


def _normalize_test_id(test_id: str) -> str:
    """Reduce a test id to the node pytest names its FAILURES section by.

    ``distributed/test_elastic_ep.py::test_scaling_uneven`` -> ``test_scaling_uneven``
    ``suite.py::TestClass::test_method`` -> ``TestClass.test_method`` (pytest joins the
    class and method with a dot in the section header).
    """
    _, separator, node = test_id.partition(".py::")
    node = node if separator else test_id
    return node.replace("::", ".")


def _match_section_name(test_id: str, section_names: list[str]) -> str | None:
    """Resolve which FAILURES section belongs to a test id.

    Both the section header and the ``FAILED`` line come from the same pytest run, so
    the header names the test's node exactly. Matching exactly (not by substring) keeps
    a test whose name prefixes another's (``test_scaling`` vs ``test_scaling_uneven``)
    from stealing its section.
    """
    node = _normalize_test_id(test_id)
    for section_name in section_names:
        if section_name == node:
            return section_name
    return None


def _get_section_body(
    test_id: str,
    section_bodies: dict[str, str],
) -> str | None:
    section_name = _match_section_name(test_id, list(section_bodies))
    return section_bodies[section_name] if section_name is not None else None


class _ExtractionResult:
    """Internal container for extraction output."""

    def __init__(self) -> None:
        self.pytest_results: list[PytestResult] = []
        self.section_bodies: dict[str, str] = {}


def _extract_pytest_failures(lines: list[str]) -> _ExtractionResult:
    """Scan lines, extract pytest sessions and per-test FAILURES section bodies."""
    result = _ExtractionResult()
    current_failures: list[FailedTest] = []
    current_section: str = ""
    section_start: int = -1

    for line_index, line in enumerate(lines):
        stripped = TIMESTAMP_RE.sub("", line).strip()

        section_match = TEST_SECTION_HEADER_RE.search(stripped)
        if section_match:
            name = section_match.group(1).strip()
            if re.search(r"[a-zA-Z0-9]", name):
                _save_section_body(
                    result, current_section, section_start, line_index, lines
                )
                current_section = name
                section_start = line_index + 1
                continue

        if _is_equals_boundary(stripped):
            _save_section_body(
                result, current_section, section_start, line_index, lines
            )
            current_section = ""
            section_start = -1

        # Assert first: the permissive inline regex would otherwise capture
        # "assert" as the class from a rewritten-assertion "- assert x == y" line.
        assert_match = FAILED_INLINE_ASSERT_RE.search(stripped)
        inline_exc_match = (
            None if assert_match else FAILED_INLINE_EXC_RE.search(stripped)
        )
        if inline_exc_match:
            failed_test = FailedTest(
                test_id=inline_exc_match.group(1),
                pytest_exception_class=inline_exc_match.group(2),
                inline_message=(inline_exc_match.group(3) or "").strip(),
            )
            current_failures.append(failed_test)
        elif assert_match:
            failed_test = FailedTest(
                test_id=assert_match.group(1),
                pytest_exception_class="AssertionError",
                inline_message=assert_match.group(2).strip(),
            )
            current_failures.append(failed_test)
        else:
            bare_failed_match = FAILED_TEST_RE.search(stripped)
            if bare_failed_match:
                failed_test = FailedTest(
                    test_id=bare_failed_match.group(1),
                )
                current_failures.append(failed_test)

        if PYTEST_SUMMARY_RE.search(stripped):
            # Record the summary's count.
            expected_count = _parse_summary_count(line.strip())
            result.pytest_results.append(
                PytestResult(
                    test_failures=current_failures,
                    pytest_summary=line.strip(),
                    expected_test_failure_count=expected_count,
                )
            )
            current_failures = []
            current_section = ""
            section_start = -1

    if current_failures:
        result.pytest_results.append(
            PytestResult(
                test_failures=current_failures,
                expected_test_failure_count=None,
            )
        )

    return result


def _is_equals_boundary(stripped: str) -> bool:
    return len(stripped) > 20 and stripped.startswith("=") and stripped.endswith("=")


def _save_section_body(
    result: _ExtractionResult,
    section_name: str,
    start: int,
    end: int,
    lines: list[str],
) -> None:
    if not section_name or start < 0:
        return
    body_lines = []
    for i in range(start, end):
        cleaned = TIMESTAMP_RE.sub("", lines[i]).rstrip()
        body_lines.append(cleaned)
    body = "\n".join(body_lines).strip()
    if body:
        result.section_bodies[section_name] = body


def _matches_infra(text: str) -> bool:
    """
    Return True for a confirmed transient (retryable) infra signature.
    """

    return any(pattern.search(text) for pattern in INFRA_PATTERNS)


def _parse_summary_count(summary: str) -> int:
    count = 0
    failed_match = SUMMARY_FAILED_COUNT_RE.search(summary)
    if failed_match:
        count += int(failed_match.group(1))
    error_match = SUMMARY_ERROR_COUNT_RE.search(summary)
    if error_match:
        count += int(error_match.group(1))
    return count
