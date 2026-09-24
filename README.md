# 한컷점검 (hancut-inspection)

> 보안검색 X-ray 사진에서, **모델이 언제 스스로 판정하고 언제 판독관에게 넘길지를 함께 학습한다.**

보안검색 판독 업무의 AI 전환 — 사람과 AI의 분업 설계. AI 캡스톤디자인 1인 프로젝트.

**2026-09-24 범위 전환(#12).** 소방시설·소화기 주제에서 X-ray 보안검색으로 옮겼습니다.
학교 AWS 계정이 `us-east-1` 고정이라 AI Hub 데이터를 쓸 수 없고, Weqaa는 라벨 한계가 남았습니다.
근거는 [D9](docs/decisions.md), 이전 문서는 `docs/archive-*/`에 있습니다.

## 무엇이 기여인가

탐지도 분류도 포화 영역입니다. **새 탐지기도, 새 손실 함수도 만들지 않습니다.**
만드는 것은 그 위에 얹히는 **위임 정책**입니다.

기존 방식은 임계값을 스윕해서 자릅니다. Learning to Defer는 사람의 정확도와 재검 비용을
손실 안에 넣어 **예측기와 위임기를 함께 학습**합니다. 방법론은 확립돼 있어 가져다 쓰고
(Mozannar & Sontag 2020, Verma & Nalisnick 2022, Mao 외 2024), 비어 있는 건 X-ray 보안검색에의 적용입니다.

1. 같은 **위임 예산**에서 학습된 위임기가 고정 임계값보다 놓침을 줄이는가
2. **판독관을 어떻게 가정하느냐가 위임 규칙을 얼마나 바꾸는가** — 가장 흥미로운 질문
3. 장비가 바뀌면 위임기를 다시 학습해야 하는가

## 구조

| 경로 | 역할 | 스택 |
|------|------|------|
| `ml/` | 데이터 준비, 판정 구간·임계값 탐색, 실험 | Python 3.11 |
| `server/` | 판정 API | FastAPI · Docker |
| `app/` | 판독 보조 앱 — 판정 확인·정정 | Flutter |
| `docs/` | 실행 계획, 실험 설계, 데이터, 백로그 | — |

## 빠른 시작

```bash
cd ml && pip install -r requirements.txt && pytest
cd server && pip install -r requirements.txt && pytest
uvicorn app.main:app --reload            # http://localhost:8000/health
cd app && flutter pub get && flutter test
```

## 핵심 설계

- **예측기 단독 성능은 결론이 아니다.** 주 지표는 **위임 예산별 시스템 놓침률** — 사람과 모델을 합친 결과다.
- **기존 3구간은 버리지 않는다.** `ml/hancut/eval/zones.py`가 이겨야 할 기준선(E2)이 된다.
- **판독관은 시뮬레이션이고, 그것을 숨기지 않는다.** PIDray의 easy/hard/hidden 덕에 균일 노이즈가 아닌
  **난이도 조건부 판독관**을 만들 수 있다. 가정을 흔드는 것 자체가 실험(E4)이다.
- **LLM에는 사진을 주지 않는다.** 판정은 비전 모델이, LLM은 구조화된 판정 결과만 받는다.
- **규정 수치를 단정하지 않는다.** TSA/ECAC의 요구 수치는 비공개다.

코드의 `facility`·`compliant` 식별자는 소방시설 시절 이름입니다. 그룹 키로는 동작하며 #13에서 정리합니다.

## 데이터 주의

**PIDray만 씁니다.** `pidray.zip` 11GB, Google Drive에서 로그인 없이 받아집니다(확인 완료).
SIXray는 비Baidu 미러가 요금 미납으로 정지돼 **받을 방법이 없습니다**.
**학술 목적 한정, 재배포 금지**이며 국외 반출 조항은 없어 `us-east-1`에서 무관합니다. 원본·전처리 이미지·모델 파일은 `data/`, `models/`에 두고
**저장소에 올리지 않습니다**(`.gitignore`). 조건은 [데이터](docs/data.md) 참고.

## 문서

| 문서 | 내용 |
|------|------|
| [실행 계획](docs/plan.md) | 12주 로드맵 (1인), 실험 E1–E6, 판독관 시뮬레이션 |
| [실험 설계](docs/experiments.md) | 질문·방법·지표 |
| [실험 수행 규칙](docs/experiment-protocol.md) | 한 번에 하나만 바꾸기, 결과 기록 형식 |
| [결정 기록](docs/decisions.md) | 왜 그렇게 했는지 — D1~D10 |
| [데이터](docs/data.md) | 출처·이용 조건·받은 뒤 확인할 것 |
| [개발 환경](docs/setup.md) | 설치·실행·테스트 |
| [백로그](docs/backlog.md) | 이슈 연결 |
| [진행 기록](docs/progress.md) | 주차별 한 줄 |
| [작업 방식](CONTRIBUTING.md) | 이슈 → 브랜치 → PR, 커밋 규칙 |
| [리뷰 가이드](AGENTS.md) | PR에서 확인할 것 |
