#!/bin/bash
# DPO 학습, 모델 배포, FastAPI 서버 실행을 자동화하는 스크립트

set -e # 오류 발생 시 스크립트 즉시 중단

echo "🚀 LabNote AI 전체 파이프라인을 시작합니다."

# --- 1. DPO 학습 실행 ---
echo "--- Step 1/3: DPO 모델 학습을 시작합니다 ---"
python scripts/run_dpo_training.py

# --- 2. 모델 배포 스크립트 실행 ---
echo "--- Step 2/3: 학습된 모델을 Ollama에 배포합니다 ---"
sh scripts/deploy_model.sh

# --- 3. FastAPI 서버 실행 ---
echo "--- Step 3/3: FastAPI 서버를 실행합니다 ---"

# 기존 uvicorn 프로세스 종료
echo ">>> 실행 중인 uvicorn 프로세스를 확인하고 종료합니다."
# uvicorn 프로세스를 찾아서 종료 (host와 port를 기준으로 찾음)
if pgrep -f "uvicorn main:app --host 127.0.0.1 --port 8000" > /dev/null
then
    pgrep -f "uvicorn main:app --host 127.0.0.1 --port 8000" | xargs kill -9
    echo ">>> 기존 uvicorn 프로세스를 종료했습니다."
fi

# nohup으로 백그라운드에서 서버 재시작
echo ">>> uvicorn main:app --host 0.0.0.0 --port 8000"
echo "서버가 백그라운드에서 실행됩니다. 로그는 nohup.out 파일을 확인하세요."
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "✅ 전체 파이프라인이 완료되었습니다."
