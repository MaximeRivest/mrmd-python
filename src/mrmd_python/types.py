"""
MRP Type Definitions

Matches the MRMD Runtime Protocol specification.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Capabilities
# =============================================================================


@dataclass
class CapabilityFeatures:
    execute: bool = True
    executeStream: bool = True
    interrupt: bool = True
    complete: bool = True
    inspect: bool = True
    hover: bool = True
    variables: bool = True
    variableExpand: bool = True
    reset: bool = True
    isComplete: bool = True
    format: bool = False
    assets: bool = True


@dataclass
class Environment:
    cwd: str = ""
    executable: str = ""
    virtualenv: str | None = None


@dataclass
class Capabilities:
    runtime: str = "mrmd-python"
    version: str = ""
    languages: list[str] = field(default_factory=lambda: ["python", "py", "python3"])
    features: CapabilityFeatures = field(default_factory=CapabilityFeatures)
    defaultSession: str = "default"
    maxSessions: int = 10
    environment: Environment = field(default_factory=Environment)
    lspFallback: str | None = None


# =============================================================================
# Sessions
# =============================================================================


@dataclass
class Session:
    id: str
    language: str = "python"
    created: str = ""
    lastActivity: str = ""
    executionCount: int = 0
    variableCount: int = 0


# =============================================================================
# Execution
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
    data: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Asset:
    path: str
    url: str
    mimeType: str
    assetType: Literal["image", "html", "svg", "data", "file"]
    size: int | None = None


@dataclass
class ExecuteResult:
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: ExecuteError | None = None
    displayData: list[DisplayData] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    executionCount: int = 0
    duration: int | None = None  # milliseconds
    imports: list[str] = field(default_factory=list)


# =============================================================================
# Completion
# =============================================================================


@dataclass
class CompletionItem:
    label: str
    insertText: str | None = None
    kind: Literal[
        "variable",
        "function",
        "method",
        "property",
        "class",
        "module",
        "keyword",
        "constant",
        "field",
        "value",
    ] = "variable"
    detail: str | None = None
    documentation: str | None = None
    valuePreview: str | None = None
    type: str | None = None


@dataclass
class CompleteResult:
    matches: list[CompletionItem] = field(default_factory=list)
    cursorStart: int = 0
    cursorEnd: int = 0
    source: Literal["runtime", "lsp", "static"] = "runtime"


# =============================================================================
# Inspection
# =============================================================================


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


# =============================================================================
# Variables
# =============================================================================


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


@dataclass
class VariableDetail(Variable):
    fullValue: str | None = None
    children: list[Variable] | None = None
    methods: list[str] | None = None
    attributes: list[str] | None = None


# =============================================================================
# Code Analysis
# =============================================================================


@dataclass
class IsCompleteResult:
    status: Literal["complete", "incomplete", "invalid", "unknown"] = "unknown"
    indent: str = ""


@dataclass
class FormatResult:
    formatted: str = ""
    changed: bool = False


# =============================================================================
# Streaming Events
# =============================================================================


@dataclass
class StdinRequest:
    prompt: str = ""
    password: bool = False
    execId: str = ""


# =============================================================================
# Exceptions
# =============================================================================


class InputCancelledError(Exception):
    """Raised when user cancels input request."""
    pass
