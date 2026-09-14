# 한컷점검 (hancut-inspection)

> 소화기·감지기·유도등을 찍으면, AI가 법 기준에 맞는지 판정하고 점검표를 채워 준다.

소방시설 자체점검 업무의 AI 전환 — 사진 기반 준수 판정과 점검표 자동 작성. AI 캡스톤디자인 프로젝트.

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

- **성능은 정확도 하나로 말하지 않는다.** 판정을 `자동 준수 / 확인 필요 / 자동 미준수` 세 구간으로 나누고, **미준수 놓침률 상한을 먼저 정한 뒤** 점검원 확인 비율을 최소화한다 → `ml/hancut/eval/zones.py`
- **LLM에는 사진을 주지 않는다.** 판정은 비전 모델이, LLM은 구조화된 판정 결과만 받아 점검표 문장을 쓴다.
- **최종 판정은 점검원이 한다.** AI는 확인할 양을 줄이는 보조다.
- **원본이 아니라 640px 서브셋으로 학습한다.** 장당 평균 1.58MB 원본을 한 번 축소해 비용과 시간을 줄인다 → `docs/data.md`

## 데이터 주의

AI Hub **다중밀집시설 및 주거시설 화재 안전 데이터**를 사용한다. 원본·전처리 이미지·모델 파일은 `data/`, `models/`에 두고 **저장소에 올리지 않는다** (`.gitignore` 처리). 이용 조건은 `docs/data.md` 참고.

## 문서

- [실행 계획 · 16주 로드맵](docs/plan.md)
- [실험 설계 E1–E7](docs/experiments.md)
- [데이터 · 서브셋 · 전처리](docs/data.md)
- [W1–W2 백로그](docs/backlog.md)
