# 🧪 프로젝트 테스트 가이드

이 문서는 `labnote-ai-backend` 프로젝트의 품질을 보장하기 위해 작성된 다양한 테스트들을 실행하고 관리하는 방법을 안내합니다. 자동화된 테스트는 코드 변경 시 발생할 수 있는 잠재적 문제를 사전에 발견하고, 시스템의 안정성을 유지하는 데 중요한 역할을 합니다.

## 목차
1. 테스트 환경 설정
2. 자동화된 단위/통합 테스트 실행
3. 테스트 커버리지 측정
4. 수동 E2E 테스트: 릴리스 워크플로우

---

## 1. 테스트 환경 설정

테스트를 실행하기 위해 필요한 개발용 의존성 패키지를 설치해야 합니다.

1.  프로젝트의 Python 가상 환경이 활성화되어 있는지 확인합니다.

2.  아래 명령어를 실행하여 `pytest`를 포함한 테스트 관련 패키지들을 설치합니다.

    ```bash
    pip install -r requirements-dev.txt
    ```

    `requirements-dev.txt` 파일에는 다음 패키지들이 포함되어 있습니다:
    - `pytest`: Python 테스트 프레임워크
    - `pytest-asyncio`: `asyncio` 기반 코드를 테스트하기 위한 플러그인
    - `httpx`: FastAPI 테스트 클라이언트가 사용하는 HTTP 클라이언트
    - `pytest-cov`: 테스트 커버리지를 측정하는 플러그인

---

## 2. 자동화된 단위/통합 테스트 실행

프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 모든 자동화 테스트를 수행할 수 있습니다.

```bash
pytest -v
```

`-v` 옵션은 각 테스트 케이스의 실행 결과를 상세하게 보여줍니다. 모든 테스트가 `PASSED`로 표시되면, 핵심 기능들이 정상적으로 작동함을 의미합니다.

### 테스트 대상 파일

현재 테스트 스위트는 `tests/` 디렉토리 내의 다음 파일들로 구성됩니다:

*   `tests/test_main_api.py`:
    *   FastAPI의 주요 API 엔드포인트(`/record_preference`, `/api/feedback_metrics` 등)의 동작을 검증합니다.
    *   사용자 피드백이 데이터베이스에 올바르게 저장되는지 확인합니다.
    *   날짜 기반 필터링 기능의 정확성을 테스트합니다.

*   `tests/test_dpo_git_push.py`:
    *   `/record_preference` API 호출 시, DPO 데이터를 Git 리포지토리로 푸시하는 로직을 검증합니다.
    *   실제 Git 명령을 실행하는 대신, 관련 함수들이 올바른 순서와 인자로 호출되는지 **모킹(mocking)**을 통해 확인합니다.

*   `tests/test_evaluate_model.py`:
    *   `scripts/evaluate_model.py`의 모델 평가 파이프라인 전체 로직을 검증합니다.
    *   LLM API 호출을 모킹하여, 평가 결과가 로그 파일과 데이터베이스에 정확히 기록되는지 테스트합니다.

---

## 3. 테스트 커버리지 측정

테스트 커버리지는 작성된 테스트 코드가 실제 운영 코드의 몇 퍼센트를 실행하는지 나타내는 지표입니다. 이를 통해 테스트가 누락된 부분을 파악할 수 있습니다.

1.  **터미널에서 커버리지 리포트 확인**:

    아래 명령어를 실행하면 `main.py`와 `scripts/` 디렉토리 내 파일들에 대한 테스트 커버리지를 측정하고, 테스트되지 않은 코드 라인을 터미널에 직접 표시해 줍니다.

    ```bash
    pytest --cov=main --cov=scripts --cov-report term-missing
    ```

2.  **HTML 리포트 생성**:

    더 시각적이고 상세한 분석을 위해 HTML 형식의 리포트를 생성할 수 있습니다.

    ```bash
    pytest --cov=main --cov=scripts --cov-report html
    ```

    명령 실행 후, 프로젝트 루트에 `htmlcov/` 디렉토리가 생성됩니다. `htmlcov/index.html` 파일을 웹 브라우저로 열면 파일별 커버리지 현황과 테스트되지 않은 코드 라인을 시각적으로 확인할 수 있습니다.

---

## 4. 수동 E2E 테스트: 릴리스 워크플로우

GitHub Actions를 이용한 자동 버전 관리 및 릴리스 기능은 실제 GitHub 환경에서 테스트해야 합니다.

1.  **테스트 브랜치 생성**:
    `main` 브랜치에서 새로운 테스트용 브랜치를 생성합니다.
    ```bash
    git checkout -b feature/test-release-workflow
    ```

2.  **코드 수정 및 커밋**:
    간단한 코드 수정(예: 주석 추가) 후, **Conventional Commits** 형식에 맞는 커밋 메시지로 커밋합니다.
    *   **Patch 버전 상승 테스트**: `git commit -m "fix: A small bug in documentation"`
    *   **Minor 버전 상승 테스트**: `git commit -m "feat: Add a new comment to a file"`

3.  **Pull Request 및 병합**:
    - 테스트 브랜치를 GitHub에 푸시합니다: `git push origin feature/test-release-workflow`
    - GitHub 웹사이트에서 `main` 브랜치를 대상으로 Pull Request(PR)를 생성합니다.
    - PR이 `main` 브랜치에 병합(Merge)되도록 합니다.

4.  **결과 확인**:
    - **Actions 탭**: 리포지토리의 `Actions` 탭으로 이동하여 `Release and Publish` 워크플로우가 성공적으로 실행되었는지 확인합니다.
    - **Releases 탭**: `Releases` 탭에서 커밋 메시지에 따라 버전이 올바르게 상승했는지(예: `v2.5.0` -> `v2.5.1`), 그리고 해당 버전의 릴리스와 태그가 `CHANGELOG.md` 내용과 함께 생성되었는지 확인합니다.