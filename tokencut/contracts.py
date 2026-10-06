"""Shared data formats for Iteration 1.

Every module talks to the others ONLY through these classes:

    ingestion.parse_repo()        -> list[Chunk]
    graph.build_graph(chunks)     -> CodeGraph
    trimming.generate_variants()  -> list[ContextVariant]

RULE: do not change this file without a Pull Request approved by the other two members.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    MODULE = "module"      # module-level code (imports, constants, top-level statements)
    CLASS = "class"
    FUNCTION = "function"  # top-level function
    METHOD = "method"      # function defined inside a class


class Chunk(BaseModel):
    """One piece of code the trimmer can keep or drop."""

    id: str = Field(description="Stable ID: '<file_path>::<qualified_name>', e.g. 'shop/cart.py::Cart.total'")
    file_path: str = Field(description="Path relative to repo root, always with forward slashes")
    qualified_name: str = Field(description="e.g. 'Cart.total'; '<module>' for module-level chunks")
    kind: ChunkKind
    start_line: int = Field(ge=1, description="1-based, inclusive")
    end_line: int = Field(ge=1, description="1-based, inclusive")
    code: str
    signature: str = Field(default="", description="e.g. 'def total(self) -> float:'; empty for module chunks")
    docstring: str = ""
    token_count: int = Field(ge=0, description="Counted with tokencut.tokens.count_tokens")
    file_hash: str = Field(description="sha256 of the whole file, used for caching / incremental updates")


class EdgeKind(str, Enum):
    CONTAINS = "contains"  # class -> method, module -> function/class
    IMPORTS = "imports"    # module chunk -> module chunk of the imported file
    CALLS = "calls"        # function/method -> function/method it calls (best guess)
    INHERITS = "inherits"  # class -> base class


class Edge(BaseModel):
    source: str = Field(description="Chunk id")
    target: str = Field(description="Chunk id")
    kind: EdgeKind
    resolved: bool = Field(default=True, description="False if the target was a heuristic guess")


class CodeGraph(BaseModel):
    chunk_ids: list[str]
    edges: list[Edge]


class Tier(str, Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    FULL = "full"


# TODO(team decision, see docs/DECISIONS.md #4): fraction of the full context's tokens per tier.
TIER_BUDGET_FRACTION: dict[Tier, float] = {
    Tier.AGGRESSIVE: 0.15,
    Tier.MODERATE: 0.35,
    Tier.CONSERVATIVE: 0.60,
    Tier.FULL: 1.00,
}


class ScoredChunk(BaseModel):
    chunk_id: str
    score: float = Field(description="Final relevance score, higher = more relevant")
    structural_score: float = 0.0
    text_score: float = 0.0


class ContextVariant(BaseModel):
    tier: Tier
    budget_tokens: int
    kept_chunk_ids: list[str] = Field(description="Sorted by file path, then start line")
    skeleton_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunks kept only as signature + docstring (see DECISIONS.md #5)",
    )
    token_count: int
    rendered_text: str = Field(description="The text that would be sent to the model")
