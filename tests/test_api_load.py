import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from tests.test_inference_api import TOKEN, headers
from visualguard.api import create_app


def test_backpressure_rejects_parallel_frames(fake_model, png):
    detector, meta = fake_model
    started, release = threading.Event(), threading.Event()
    predict = detector.predict

    def slow(images):
        started.set()
        if not release.wait(5):
            raise RuntimeError("test synchronization timeout")
        return predict(images)

    detector.predict = slow
    with (
        TestClient(create_app(detector, meta, token=TOKEN)) as client,
        ThreadPoolExecutor(1) as pool,
    ):
        future = pool.submit(client.post, "/predict", content=png, headers=headers("first"))
        assert started.wait(5)
        try:
            assert (
                client.post("/predict", content=png, headers=headers("second")).status_code == 429
            )
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200


def test_rate_limit_is_bounded(fake_model, png):
    with TestClient(create_app(*fake_model, token=TOKEN)) as client:
        for _ in range(60):
            assert client.post("/predict", content=png, headers=headers()).status_code == 200
        assert client.post("/predict", content=png, headers=headers()).status_code == 429
