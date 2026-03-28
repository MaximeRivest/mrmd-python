"""
MRP Type Definitions — 0.4.0-draft

Two layers:
- Engine types (dataclasses): used internally by IPythonWorker and subprocess layer.
  These are stable and should not change when the protocol evolves.
- Protocol types (dataclasses): match the MRP 0.4.0 spec schemas.
  These are what the MCP tools return.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Exceptions
# =============================================================================


class InputCancelledError(Exception):
    """Raised when user cancels input request."""
    pass


class StdinNotAllowedError(Exception):
    """Raised when code requests stdin in a non-interactive execution context."""
    pass


# =============================================================================
# Engine Types — used by worker.py and subprocess_manager.py
#
# These are the internal data containers. The worker produces these.
# The service layer converts them into protocol types.
# =============================================================================


@dataclass
class ExecuteError:
    type: str
    message: str
    traceback: list[str] = field(default_factory=list)
    line: int | None = None
    column: int | None = None


@dataclass
class DisplayData:
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Asset:
    id: str
    url: str
    mimeType: str
    assetType: Literal["image", "html", "svg", "data", "file"]
    size: int | None = None


@dataclass
class ExecuteResult:
    """Engine-level execution result. Produced by IPythonWorker."""
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: ExecuteError | None = None
    displayData: list[DisplayData] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    executionCount: int = 0
    stateRevision: int = 0
    duration: int | None = None  # milliseconds
    imports: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CompletionItem:
    label: str
    insertText: str | None = None
    kind: Literal[
        "variable", "function", "method", "property", "class",
        "module", "keyword", "constant", "field", "value",
    ] = "variable"
    detail: str | None = None
    documentation: str | None = None
    valuePreview: str | None = None
    type: str | None = None
    sortText: str | None = None


@dataclass
class CompleteResult:
    matches: list[CompletionItem] = field(default_factory=list)
    cursorStart: int = 0
    cursorEnd: int = 0
    source: Literal["runtime", "lsp", "static"] = "runtime"


@dataclass
class InspectResult:
    found: bool = False
    source: Literal["runtime", "lsp", "static"] = "runtime"
    name: str | None = None
    kind: Literal[
        "variable", "function", "class", "module", "method", "property"
    ] | None = None
    type: str | None = None
    signature: str | None = None
    docstring: str | None = None
    sourceCode: str | None = None
    file: str | None = None
    line: int | None = None
    value: str | None = None
    children: int | None = None


@dataclass
class HoverResult:
    found: bool = False
    name: str | None = None
    type: str | None = None
    value: str | None = None
    signature: str | None = None
    docstring: str | None = None


@dataclass
class Variable:
    name: str
    type: str
    value: str
    size: str | None = None
    expandable: bool = False
    shape: list[int] | None = None
    dtype: str | None = None
    length: int | None = None
    keys: list[str] | None = None


@dataclass
class VariablesResult:
    variables: list[Variable] = field(default_factory=list)
    count: int = 0
    truncated: bool = False
    warnings: list[dict[str, str]] = field(default_factory=list)


@dataclass
class VariableDetail(Variable):
    fullValue: str | None = None
    children: list[Variable] | None = None
    methods: list[str] | None = None
    attributes: list[str] | None = None


@dataclass
class IsCompleteResult:
    status: Literal["complete", "incomplete", "invalid", "unknown"] = "unknown"
    indent: str = ""


@dataclass
class FormatResult:
    formatted: str = ""
    changed: bool = False


@dataclass
class HistoryEntry:
    historyIndex: int
    code: str


@dataclass
class HistoryResult:
    entries: list[HistoryEntry] = field(default_factory=list)
    hasMore: bool = False


@dataclass
class StdinRequest:
    prompt: str = ""
    password: bool = False
    execId: str = ""


# =============================================================================
# Protocol Types — MRP 0.4.0-draft
#
# These match the spec schemas in mrp-openapi.yaml.
# The service layer produces these from engine types.
# =============================================================================


@dataclass
class Warning:
    code: str
    message: str


@dataclass
class CapabilityFeatures:
    execute: bool = True
    executionEvents: bool = True
    input: bool = True
    cancelExecution: bool = True
    complete: bool = True
    inspect: bool = True
    hover: bool = True
    variables: bool = True
    variableExpand: bool = True
    reset: bool = True
    isComplete: bool = True
    format: bool = False
    history: bool = True
    assets: bool = True


@dataclass
class Environment:
    cwd: str = ""
    executable: str = ""
    virtualenv: str | None = None


@dataclass
class Capabilities:
    protocol: str = "mrp"
    protocolVersion: str = "0.4.0-draft"
    runtime: str = "rat-py"
    version: str = "0.5.0"
    languages: list[str] = field(default_factory=lambda: ["python", "py", "python3"])
    features: CapabilityFeatures = field(default_factory=CapabilityFeatures)
    environment: Environment = field(default_factory=Environment)
    lspFallback: str | None = None


@dataclass
class Health:
    status: str = "ok"
    state: str = "idle"  # idle, executing, waiting_input
    runtime: str = "rat-py"
    currentExecutionId: str | None = None
    executionCount: int = 0
    stateRevision: int = 0
    uptimeSeconds: int = 0


@dataclass
class InputRequestInfo:
    inputRequestId: str
    prompt: str = ""
    password: bool = False


@dataclass
class ExecutionState:
    executionId: str
    clientExecutionId: str | None = None
    status: Literal["working", "input_required", "completed", "failed", "cancelled"] = "working"
    statusMessage: str | None = None
    createdAt: str = ""
    lastUpdatedAt: str = ""
    pollIntervalMs: int | None = 500
    ttlMs: int | None = 3600000
    currentSeq: int = 0
    stateRevision: int = 0
    inputRequest: InputRequestInfo | None = None
    warnings: list[Warning] = field(default_factory=list)


@dataclass
class ExecutionEvent:
    seq: int
    timestamp: str
    type: Literal["stdout", "stderr", "display", "asset", "warning", "input_requested"]
    # type-specific fields — exactly one should be set
    content: str | None = None            # stdout, stderr
    displayData: DisplayData | None = None  # display
    asset: Asset | None = None             # asset
    warning: Warning | None = None          # warning
    inputRequest: InputRequestInfo | None = None  # input_requested


@dataclass
class ExecutionEventsResponse:
    executionId: str
    events: list[ExecutionEvent] = field(default_factory=list)
    nextSeq: int = 1
    done: bool = False


@dataclass
class ExecutionResult:
    """Protocol-level final execution result."""
    executionId: str
    status: Literal["completed", "failed", "cancelled"] = "completed"
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: ExecuteError | None = None
    displayData: list[DisplayData] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    executionCount: int = 0
    stateRevision: int = 0
    durationMs: int | None = None
    imports: list[str] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
