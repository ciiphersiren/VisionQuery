import pytest
from fastapi.testclient import TestClient

from skyquery.api.server import SAMPLES_DIR, app

c = TestClient(app)


def test_health_and_plan():
    assert c.get("/api/health").json()["ok"]
    p = c.post("/api/plan", json={"query": "any red buses on the left?"}).json()
    assert p["oov"][0]["nearest"] == ["bus"] and p["region"] == "left half"


def test_unknown_image_is_404_not_a_path():
    assert c.post("/api/ask", json={"image_id": "/etc/passwd", "query": "how many cars"}).status_code == 404
    assert c.post("/api/detect", json={"image_id": "pyproject.toml"}).status_code == 404
    assert c.get("/api/samples/..%2Fpyproject.toml").status_code == 404


def test_bad_pipeline():
    assert c.post("/api/ask", json={"image_id": "x", "query": "cars", "pipeline": "fi9999"}).status_code == 400


@pytest.mark.slow
def test_end_to_end_sample():
    name = sorted(SAMPLES_DIR.glob("*.jpg"))[0].name
    r = c.post(f"/api/samples/{name}/load").json()
    a = c.post("/api/ask", json={"image_id": r["image_id"], "query": "how many cars?"}).json()
    assert isinstance(a["answer"]["value"], int)
    assert len(a["answer"]["evidence"]) == a["answer"]["value"]


def test_meta_lists_models_and_rejects_bad_choices():
    m = c.get("/api/meta").json()
    ids = {x["id"]: x for x in m["models"]}
    assert m["default_model"] in ids and ids[m["default_model"]]["available"]
    assert ids["yolo26m_tile_uavdt"]["classes"] == ["car", "truck", "bus"]
    assert c.post("/api/ask", json={"image_id": "x", "query": "cars", "model": "nope"}).status_code == 400
    assert c.post("/api/ask", json={"image_id": "x", "query": "cars", "model": "yolo26s_visdrone", "pipeline": "sliced"}).status_code == 400
    missing = [x for x in m["models"] if not x["available"]]
    if missing:
        r = c.post("/api/ask", json={"image_id": "x", "query": "cars", "model": missing[0]["id"]})
        assert r.status_code == 409 and missing[0]["missing"][0] in r.json()["detail"]
