"""Member B — HTTP API."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_variants_on_sample_repo(sample_repo_path):
    response = client.post(
        "/context/variants",
        json={"repo_path": str(sample_repo_path), "query": "Cart.total is wrong"},
    )
    assert response.status_code == 200
    body = response.json()
    tiers = [v["tier"] for v in body["variants"]]
    assert tiers == ["aggressive", "moderate", "conservative", "full"]
    assert body["full_tokens"] == body["variants"][-1]["token_count"]
    assert all("rendered_text" in v for v in body["variants"])


def test_variants_without_text(sample_repo_path):
    response = client.post(
        "/context/variants",
        json={"repo_path": str(sample_repo_path), "query": "refund", "include_text": False},
    )
    assert response.status_code == 200
    assert all("rendered_text" not in v for v in response.json()["variants"])


def test_bad_path_is_400(tmp_path):
    response = client.post(
        "/context/variants", json={"repo_path": str(tmp_path / "nope"), "query": "x"}
    )
    assert response.status_code == 400


def test_folder_without_python_is_422(tmp_path):
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    response = client.post("/context/variants", json={"repo_path": str(tmp_path), "query": "x"})
    assert response.status_code == 422
    assert "no Python files" in response.json()["detail"]
