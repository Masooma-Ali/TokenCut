"""HTTP endpoint for the Iteration 1 pipeline (IMPLEMENTATION_PLAN B7).

Run:  uvicorn api.main:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tokencut.contracts import Tier
from tokencut.graph.builder import build_graph
from tokencut.ingestion.parser import parse_repo
from tokencut.trimming.variants import generate_variants

app = FastAPI(title="TokenCut", version="0.1.0")


class VariantsRequest(BaseModel):
    repo_path: str
    query: str
    include_text: bool = True
    full_cap_tokens: int | None = None  # None = keep the whole repo as "full" (gateway mode)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/context/variants")
def context_variants(request: VariantsRequest) -> dict:
    """Returns {"full_tokens": int, "variants": [ContextVariant, ...]} in tier order.

    With include_text=false each variant omits rendered_text (smaller response for dashboards).
    """
    repo = Path(request.repo_path)
    if not repo.is_dir():
        raise HTTPException(
            status_code=400, detail=f"repo_path is not a folder: {request.repo_path}"
        )
    chunks = parse_repo(repo)
    if not chunks:
        raise HTTPException(status_code=422, detail="no Python files found in repo_path")

    graph = build_graph(chunks)
    variants = generate_variants(
        chunks, graph, request.query, full_cap_tokens=request.full_cap_tokens
    )
    full_tokens = next(v.token_count for v in variants if v.tier == Tier.FULL)

    payload = []
    for v in variants:
        data = v.model_dump(mode="json")
        if not request.include_text:
            del data["rendered_text"]
        payload.append(data)
    return {"full_tokens": full_tokens, "variants": payload}
