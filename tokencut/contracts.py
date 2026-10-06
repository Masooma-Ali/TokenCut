"""Shared data formats for Iteration 1.

Every module talks to the others ONLY through these classes:

    ingestion.parse_repo()        -> list[Chunk]
    graph.build_graph(chunks)     -> CodeGraph
    trimming.generate_variants()  -> list[ContextVariant]

RULE: do not change this file without a Pull Request approved by the other two members.
See docs/DECISIONS.md for the reasoning behind every rule below.
"""

from enum import Enum

from pydantic import BaseModel, Field

# ---------- Shared settings (DECISIONS.md #4, #12) ----------

EVAL_FULL_CONTEXT_CAP_TOKENS = 32_000   # "full" context size in evaluation (tunable)
AGGRESSIVE_MIN_TOKENS = 1_000           # aggressive tier is never smaller than this
MAX_FILE_BYTES = 1_000_000              # files larger than this are skipped
SKIP_DIR_NAMES = {"migrations", "vendor", ".git", ".venv", "venv", "__pycache__", "node_modules"}


# ---------- Chunks (DECISIONS.md #2) ----------

class ChunkKind(str, Enum):
    MODULE = "module"      # module-level code: imports, constants, top-level statements (no defs)
    CLASS = "class"        # class header + class-level attributes ONLY (methods are separate chunks)
    FUNCTION = "function"  # top-level function (nested functions stay inside it)
    METHOD = "method"      # function defined directly inside a class


class Chunk(BaseModel):
    """One piece of code the trimmer can keep, skeletonize, or drop."""

    id: str = Field(
        description="'<file_path>::<qualified_name>', e.g. 'shop/cart.py::Cart.total'. "
        "A repeated name in the same file gets a suffix: '...::Cart.price#2'"
    )
    file_path: str = Field(description="Relative to repo root, ALWAYS forward slashes")
    qualified_name: str = Field(description="e.g. 'Cart.total'; '<module>' for module-level chunks")
    kind: ChunkKind
    parent_id: str | None = Field(default=None, description="For methods: id of their class chunk")
    start_line: int = Field(ge=1, description="1-based, inclusive")
    end_line: int = Field(ge=1, description="1-based, inclusive")
    code: str
    signature: str = Field(default="", description="e.g. 'def total(self) -> float:'; empty for module chunks")
    docstring: str = ""
    token_count: int = Field(ge=0, description="Counted with tokencut.tokens.count_tokens")
    file_hash: str = Field(description="sha256 of the whole file, for caching / incremental updates")


# ---------- Graph (DECISIONS.md #7) ----------

class EdgeKind(str, Enum):
    CONTAINS = "contains"  # class -> method
    IMPORTS = "imports"    # module chunk -> module chunk of the imported file
    CALLS = "calls"        # function/method -> function/method it calls
    INHERITS = "inherits"  # class -> base class


class Edge(BaseModel):
    source: str = Field(description="Chunk id")
    target: str = Field(description="Chunk id (must exist in the graph)")
    kind: EdgeKind
    resolved: bool = Field(default=True, description="False if found by unique-name guess (rule 4)")


class CodeGraph(BaseModel):
    chunk_ids: list[str]
    edges: list[Edge]


# ---------- Tiers and variants (DECISIONS.md #4, #5) ----------

class Tier(str, Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    FULL = "full"


# Fraction of the "full" context's tokens per tier (tunable).
TIER_BUDGET_FRACTION: dict[Tier, float] = {
    Tier.AGGRESSIVE: 0.15,
    Tier.MODERATE: 0.35,
    Tier.CONSERVATIVE: 0.60,
    Tier.FULL: 1.00,
}

# Tiers in which connected dropped chunks are kept as skeletons (signature + docstring).
SKELETON_TIERS: set[Tier] = {Tier.AGGRESSIVE, Tier.MODERATE}


class ScoredChunk(BaseModel):
    chunk_id: str
    score: float = Field(description="Final relevance score, higher = more relevant")
    structural_score: float = 0.0
    text_score: float = 0.0


class ContextVariant(BaseModel):
    tier: Tier
    budget_tokens: int
    kept_chunk_ids: list[str] = Field(description="Full code kept; sorted by file path, then start line")
    skeleton_chunk_ids: list[str] = Field(
        default_factory=list, description="Kept as signature + docstring only; same sort order"
    )
    token_count: int = Field(description="Includes skeleton tokens; must be <= budget_tokens")
    rendered_text: str = Field(description="The text that would be sent to the model")