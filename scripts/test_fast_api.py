import pytest
from fastapi.testclient import TestClient
from app.api.v1 import predict

client = TestClient(predict.app)

def test_predict_endpoint():
    data = {
        "comments": ["This is a great product!", "Not worth the money.", "It's okay."]
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_with_timestamps_endpoint():
    data = {
        "comments": [
            {"text": "This is fantastic!", "timestamp": "2024-10-25 10:00:00"},
            {"text": "Could be better.", "timestamp": "2024-10-26 14:00:00"}
        ]
    }
    response = client.post("/predict_with_timestamps", json=data)
    assert response.status_code == 200
    assert all("sentiment" in item for item in response.json())

def test_generate_chart_endpoint():
    data = {
        "sentiment_counts": {"1": 5, "0": 3, "-1": 2}
    }
    response = client.post("/generate_chart", json=data)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"

def test_generate_wordcloud_endpoint():
    data = {
        "comments": ["Love this!", "Not so great.", "Absolutely amazing!", "Horrible experience."]
    }
    response = client.post("/generate_wordcloud", json=data)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"

def test_generate_trend_graph_endpoint():
    data = {
        "sentiment_data": [
            {"timestamp": "2024-10-01", "sentiment": 1},
            {"timestamp": "2024-10-02", "sentiment": 0},
            {"timestamp": "2024-10-03", "sentiment": -1}
        ]
    }
    response = client.post("/generate_trend_graph", json=data)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"

def test_fetch_youtube_comments_endpoint():
    data = {"video_id": "eKHoLpi2ey4"}
    response = client.post("/fetch_comments", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)