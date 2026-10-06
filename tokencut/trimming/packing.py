"""Choose the 'full' pool and pack the four nested tiers (IMPLEMENTATION_PLAN C3, C4).

Tiers are packed aggressive -> moderate -> conservative -> full. Each tier starts
from everything the previous tier contained, so kept sets are nested and token
counts never decrease. Each tier is rendered and its real token count checked
against the budget; if the estimate was too low, the lowest-ranked chunk added
in this tier is removed and the text re-rendered until it fits.
"""

from dataclasses import dataclass

from tokencut.contracts import (
    AGGRESSIVE_MIN_TOKENS,
    SKELETON_TIERS,
    TIER_BUDGET_FRACTION,
    Chunk,
    ChunkKind,
    ScoredChunk,
    Tier,
)
from tokencut.graph.queries import GraphIndex
from tokencut.tokens import count_tokens
from tokencut.trimming.render import file_header, render_variant, skeleton_text

TIER_ORDER = [Tier.AGGRESSIVE, Tier.MODERATE, Tier.CONSERVATIVE, Tier.FULL]
SEPARATOR_TOKENS = 1  # the blank line between two rendered chunks, roughly one token
_SKELETON_KINDS = (ChunkKind.CLASS, ChunkKind.FUNCTION, ChunkKind.METHOD)


@dataclass
class PackedTier:
    tier: Tier
    budget_tokens: int
    kept: set[str]
    skeletons: set[str]
    rendered_text: str
    token_count: int


def _rank_files(chunks: list[Chunk], scores: list[ScoredChunk]) -> list[str]:
    best: dict[str, float] = {}
    score_by_id = {s.chunk_id: s.score for s in scores}
    for c in chunks:
        best[c.file_path] = max(best.get(c.file_path, float("-inf")), score_by_id.get(c.id, 0.0))
    return sorted(best, key=lambda f: (-best[f], f))


def build_pool(
    chunks: list[Chunk], scores: list[ScoredChunk], full_cap_tokens: int | None
) -> list[Chunk]:
    """The 'full' context (DECISIONS #4).

    full_cap_tokens=None (gateway mode): every chunk. Otherwise (evaluation): whole files in
    order of their best chunk score while the total fits the cap; a file that does not fit is
    skipped and smaller ones are still tried.
    """
    if full_cap_tokens is None:
        return list(chunks)
    file_cost: dict[str, int] = {}
    for c in chunks:
        file_cost[c.file_path] = file_cost.get(c.file_path, count_tokens(file_header(c.file_path)))
        file_cost[c.file_path] += c.token_count + SEPARATOR_TOKENS
    selected: set[str] = set()
    total = 0
    for file_path in _rank_files(chunks, scores):
        if total + file_cost[file_path] <= full_cap_tokens:
            selected.add(file_path)
            total += file_cost[file_path]
    return [c for c in chunks if c.file_path in selected]


def tier_budgets(
    full_tokens: int, aggressive_min_tokens: int = AGGRESSIVE_MIN_TOKENS
) -> dict[Tier, int]:
    """Budget = fraction of full (DECISIONS #4), aggressive floor, capped at full.

    A tier's budget is never below the previous tier's: on a small repo the aggressive
    floor can exceed 35% of full, and moderate must not then get a smaller budget.
    """
    budgets = {}
    previous = 0
    for tier in TIER_ORDER:
        budget = round(TIER_BUDGET_FRACTION[tier] * full_tokens)
        if tier == Tier.AGGRESSIVE:
            budget = max(budget, aggressive_min_tokens)
        budget = min(max(budget, previous), full_tokens)
        budgets[tier] = budget
        previous = budget
    return budgets


class TierPacker:
    def __init__(self, pool: list[Chunk], scores: list[ScoredChunk], index: GraphIndex):
        self.pool = pool
        self.by_id = {c.id: c for c in pool}
        # Rank = position in score order (score desc, ties by id); pool chunks without a score go last.
        ordered = [s.chunk_id for s in scores if s.chunk_id in self.by_id]
        seen = set(ordered)
        ordered += sorted(cid for cid in self.by_id if cid not in seen)
        self.ranked = ordered
        self.rank = {cid: i for i, cid in enumerate(ordered)}
        self.index = index
        self._skeleton_tokens: dict[str, int] = {}
        self._header_tokens: dict[str, int] = {}

    # ---------- cost estimates ----------

    def _skel_cost(self, cid: str) -> int:
        if cid not in self._skeleton_tokens:
            self._skeleton_tokens[cid] = (
                count_tokens(skeleton_text(self.by_id[cid])) + SEPARATOR_TOKENS
            )
        return self._skeleton_tokens[cid]

    def _full_cost(self, cid: str) -> int:
        return self.by_id[cid].token_count + SEPARATOR_TOKENS

    def _header_cost(self, file_path: str) -> int:
        if file_path not in self._header_tokens:
            # No separator here: the newline after the header is the first chunk's separator.
            self._header_tokens[file_path] = count_tokens(file_header(file_path))
        return self._header_tokens[file_path]

    def _estimate(self, kept: set[str], skeletons: set[str]) -> int:
        files = {self.by_id[cid].file_path for cid in kept | skeletons}
        return (
            sum(self._full_cost(cid) for cid in kept)
            + sum(self._skel_cost(cid) for cid in skeletons)
            + sum(self._header_cost(f) for f in files)
        )

    def _cost_to_keep(self, group: set[str], skeletons: set[str], files: set[str]) -> int:
        """Extra tokens if `group` becomes full code (a skeleton being upgraded costs the difference)."""
        cost = 0
        new_files = set()
        for cid in group:
            cost += self._full_cost(cid)
            if cid in skeletons:
                cost -= self._skel_cost(cid)
            file_path = self.by_id[cid].file_path
            if file_path not in files:
                new_files.add(file_path)
        return cost + sum(self._header_cost(f) for f in new_files)

    def _parent(self, cid: str) -> str | None:
        parent = self.by_id[cid].parent_id
        return parent if parent in self.by_id else None

    # ---------- packing steps ----------

    def _add_full_code(self, kept: set[str], skeletons: set[str], budget: int) -> None:
        remaining = budget - self._estimate(kept, skeletons)
        files = {self.by_id[cid].file_path for cid in kept | skeletons}
        for cid in self.ranked:
            if cid in kept or self.by_id[cid].token_count == 0:
                continue  # an empty chunk (e.g. empty __init__.py) adds only a file header
            group = (self.index.closure({cid}, max_hops=1) & self.by_id.keys()) - kept
            group.add(cid)
            cost = self._cost_to_keep(group, skeletons, files)
            if cost > remaining:
                # Fall back to the chunk alone, plus what it needs to render correctly.
                group = {cid}
                parent = self._parent(cid)
                if parent is not None:
                    group.add(parent)
                module_id = self.index.module_of(cid)
                if module_id is not None and module_id in self.by_id:
                    group.add(module_id)
                group -= kept
                cost = self._cost_to_keep(group, skeletons, files)
                if cost > remaining:
                    continue  # keep going: a smaller chunk further down may still fit
            kept |= group
            skeletons -= group
            files |= {self.by_id[g].file_path for g in group}
            remaining -= cost

    def _add_skeletons(self, kept: set[str], skeletons: set[str], budget: int) -> None:
        # DECISIONS #5: only dropped chunks directly connected to kept code become skeletons.
        remaining = budget - self._estimate(kept, skeletons)
        files = {self.by_id[cid].file_path for cid in kept | skeletons}
        candidates = []
        for cid in self.ranked:
            if cid in kept or cid in skeletons or self.by_id[cid].kind not in _SKELETON_KINDS:
                continue
            if any(n in kept for n in self.index.neighbours(cid)):
                candidates.append(cid)
        for cid in candidates:
            if cid in skeletons:
                continue
            group = [cid]
            parent = self._parent(cid)
            if parent is not None and parent not in kept and parent not in skeletons:
                group.append(parent)  # a method skeleton needs its class line above it
            cost = sum(self._skel_cost(g) for g in group)
            new_files = {self.by_id[g].file_path for g in group} - files
            cost += sum(self._header_cost(f) for f in new_files)
            if cost <= remaining:
                skeletons.update(group)
                files |= new_files
                remaining -= cost

    def _render(self, kept: set[str], skeletons: set[str]) -> tuple[str, int]:
        text = render_variant(self.pool, kept, skeletons)
        return text, count_tokens(text)

    def _removable(
        self, ids: set[str], protected: set[str], kept: set[str], skeletons: set[str]
    ) -> list[str]:
        """ids that can be removed without orphaning a method (its class must stay rendered)."""
        has_children = {self._parent(cid) for cid in kept | skeletons}
        return [cid for cid in ids if cid not in protected and cid not in has_children]

    def _fit(
        self,
        kept: set[str],
        skeletons: set[str],
        base_kept: set[str],
        base_skeletons: set[str],
        budget: int,
    ) -> tuple[str, int]:
        """Remove lowest-ranked chunks added in this tier until the rendered text fits the budget.

        Everything in the previous tier (base_*) is protected; the previous tier's text fit a
        smaller budget, so this loop always ends within budget.
        """
        while True:
            text, tokens = self._render(kept, skeletons)
            if tokens <= budget:
                return text, tokens
            victims = self._removable(skeletons, base_skeletons, kept, skeletons)
            if not victims:
                victims = self._removable(kept, base_kept, kept, skeletons)
            if not victims:
                return text, tokens  # unreachable in practice: the base content already fit
            victim = max(victims, key=lambda cid: self.rank[cid])
            if victim in kept:
                kept.discard(victim)
                if victim in base_skeletons:
                    skeletons.add(victim)  # it was a skeleton before: fall back to that
            else:
                skeletons.discard(victim)

    def pack(self, budgets: dict[Tier, int]) -> list[PackedTier]:
        results: list[PackedTier] = []
        prev_kept: set[str] = set()
        prev_skeletons: set[str] = set()
        prev_text, prev_tokens = "", 0

        for tier in TIER_ORDER:
            budget = budgets[tier]
            if tier == Tier.FULL:
                kept, skeletons = set(self.by_id), set()
                text, tokens = self._render(kept, skeletons)
            else:
                kept = set(prev_kept)
                skeletons = set(prev_skeletons) if tier in SKELETON_TIERS else set()
                self._add_full_code(kept, skeletons, budget)
                if tier in SKELETON_TIERS:
                    self._add_skeletons(kept, skeletons, budget)
                # In a skeleton tier, a previous skeleton upgraded to full code may fall back to a skeleton.
                base_skeletons = prev_skeletons if tier in SKELETON_TIERS else set()
                text, tokens = self._fit(kept, skeletons, prev_kept, base_skeletons, budget)

                if tokens < prev_tokens and tier not in SKELETON_TIERS:
                    # Dropping the previous tier's skeletons made this tier smaller: bring them back.
                    skeletons = prev_skeletons - kept
                    text, tokens = self._fit(kept, skeletons, prev_kept, prev_skeletons, budget)
                if tokens < prev_tokens:
                    # Safety net (token merging can shift counts by a token or two).
                    kept, skeletons = set(prev_kept), set(prev_skeletons)
                    text, tokens = prev_text, prev_tokens

            results.append(PackedTier(tier, budget, kept, skeletons, text, tokens))
            prev_kept, prev_skeletons, prev_text, prev_tokens = kept, skeletons, text, tokens
        return results
