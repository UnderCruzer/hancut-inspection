> **2026-09-24 정정:** 소방시설 주제를 유지합니다. Weqaa를 우선 검증하며 VisA/Real-IAD 전환은 취소합니다. [현재 계획](docs/plan.md) · [라벨 조사](docs/fire-dataset-review.md) · [Windows 인수인계](docs/windows-handoff.md).

# 한컷점검 (hancut-inspection)

> 소화기 사진에서 압력계 상태·부식 의심 위치를 찾고, 담당자 확인과 점검 기록 작성을 돕는다.

소방시설 점검 업무의 AI 전환 — 사진상 상태 확인과 점검 기록 작성 보조. AI 캡스톤디자인 1인 프로젝트.

## 구조

| 경로 | 역할 | 스택 |
|------|------|------|
| `ml/` | 데이터 준비, 평가 지표, 학습·실험 | Python 3.11 |
| `server/` | 판정 API | FastAPI · Docker |
| `app/` | 점검원 앱 — 촬영, 판정 확인·정정 | Flutter |
| `docs/` | 실행 계획, 실험 설계, 데이터, 백로그 | — |

## 빠른 시작

```bash
# ML — 평가 지표와 데이터 도구
cd ml && pip install -r requirements.txt && pytest

# 서버
cd server && pip install -r requirements.txt && pytest
uvicorn app.main:app --reload            # http://localhost:8000/health

# 앱
cd app && flutter pub get && flutter test
```

## 핵심 설계

- **성능은 정확도 하나로 말하지 않는다.** 사진상 항목 판정을 `정상 후보 / 확인 필요 / 이상 의심` 세 구간으로 나누고, **미준수 놓침률 상한을 먼저 정한 뒤** 점검원 확인 비율을 최소화한다 → `ml/hancut/eval/zones.py`
- **LLM에는 사진을 주지 않는다.** 판정은 비전 모델이, LLM은 구조화된 판정 결과만 받아 점검표 문장을 쓴다.
- **최종 판정은 점검원이 한다.** AI는 확인할 양을 줄이는 보조다.
- **원본이 아니라 640px 서브셋으로 학습한다.** Weqaa v18은 640×640이며 실제 학습 시간·용량은 확보 후 측정한다 → `docs/data.md`

기존 API의 compliant 명칭은 구현 이력이며, 이번 서비스에서 법적 준수 판정을 뜻하지 않는다. 화면 용어 변경은 후속 구현이다.

## 데이터 주의

**Weqaa 소화기 상태 데이터**를 우선 검증한다. AI Hub 데이터는 국외 반출 합의가 없어 현재 미국 AWS에서 사용하지 않는다. 원본·전처리 이미지·모델 파일은 `data/`, `models/`에 두고 **저장소에 올리지 않는다** (`.gitignore` 처리). 이용 조건은 `docs/data.md` 참고.

## 문서

| 문서 | 내용 |
|------|------|
| [실행 계획](docs/plan.md) | 16주 로드맵 (1인 기준), 범위 조정 |
| [실험 설계](docs/experiments.md) | E1–E7 질문·방법·지표 |
| [실험 수행 규칙](docs/experiment-protocol.md) | 한 번에 하나만 바꾸기, 결과 기록 형식 |
| [결정 기록](docs/decisions.md) | 왜 그렇게 했는지 — D1~D7 |
| [데이터](docs/data.md) | 출처·받는 법·B안 전처리·W2 확인 항목 |
| [개발 환경](docs/setup.md) | 설치·실행·테스트, 자주 겪는 문제 |
| [백로그](docs/backlog.md) | 이슈 #1–#10 연결 |
| [진행 기록](docs/progress.md) | 주차별 한 줄 |
| [작업 방식](CONTRIBUTING.md) | 이슈 → 브랜치 → PR, 커밋 규칙 |
| [리뷰 가이드](AGENTS.md) | PR에서 확인할 것 |
