import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import json
import sqlite3
import os

TEST_DB_PATH = os.getenv("EVALUATION_DB_PATH", "./test_evaluation_results.db")

@pytest.mark.asyncio
async def test_record_preference_and_get_metrics(client: TestClient):
    """
    /record_preference로 피드백을 저장하고 /api/feedback_metrics로 조회하는 기능 테스트
    """
    # Git 관련 동작은 모킹(mocking)하여 실제 Git 명령이 실행되지 않도록 함
    with patch('main.git.Repo') as mock_repo:
        # 테스트용 요청 데이터
        test_payload = {
            "uo_id": "UHW010",
            "section": "Method",
            "chosen_original": "AI suggestion",
            "chosen_edited": "User edited suggestion",
            "rejected": ["other option 1"],
            "query": "Test query",
            "file_content": "### [UHW010 Liquid Handling]",
            "file_path": "/test/path/001_WF_Test.md",
            "supervisor_evaluations": []
        }

        # 1. 피드백 기록 API 호출
        response = client.post("/record_preference", json=test_payload)
        assert response.status_code == 204

        # 2. DB에 데이터가 올바르게 저장되었는지 직접 확인
        with sqlite3.connect(TEST_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT uo_id, section, edit_distance_ratio FROM feedback_metrics")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "UHW010"
            assert row[1] == "Method"
            assert row[2] > 0.0 # edit_distance_ratio가 계산되었는지 확인

        # 3. 피드백 조회 API 호출
        response = client.get("/api/feedback_metrics")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["uo_id"] == "UHW010"
        assert data[0]["edit_distance_ratio"] == pytest.approx(row[2])

def test_get_feedback_metrics_with_date_filter(client: TestClient):
    """
    /api/feedback_metrics의 날짜 필터링 기능 테스트
    """
    with patch('main.git.Repo'):
        # 테스트 데이터 2개 삽입
        client.post("/record_preference", json={
            "uo_id": "UHW010", "section": "Method", "chosen_original": "a", "chosen_edited": "b",
            "rejected": [], "query": "q", "file_content": "c", "file_path": "f", "supervisor_evaluations": []
        })
        # 두 번째 요청은 다른 날짜에 발생한 것처럼 가정하기 위해 DB 직접 수정
        with sqlite3.connect(TEST_DB_PATH) as conn:
            conn.execute("UPDATE feedback_metrics SET timestamp = '2023-01-01T12:00:00Z' WHERE id = 1")

        client.post("/record_preference", json={
            "uo_id": "UHW020", "section": "Reagent", "chosen_original": "c", "chosen_edited": "d",
            "rejected": [], "query": "q", "file_content": "c", "file_path": "f", "supervisor_evaluations": []
        })

    # 1. 필터링 없이 전체 조회
    response = client.get("/api/feedback_metrics")
    assert len(response.json()) == 2

    # 2. 시작일 필터링
    response = client.get("/api/feedback_metrics?start_date=2024-01-01")
    data = response.json()
    assert len(data) == 1
    assert data[0]["uo_id"] == "UHW020"

    # 3. 종료일 필터링
    response = client.get("/api/feedback_metrics?end_date=2023-12-31")
    data = response.json()
    assert len(data) == 1
    assert data[0]["uo_id"] == "UHW010"

    # 4. 기간 필터링 (결과 없음)
    response = client.get("/api/feedback_metrics?start_date=2025-01-01")
    assert len(response.json()) == 0