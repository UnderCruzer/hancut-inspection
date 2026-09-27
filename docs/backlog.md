# 백로그

2026-09-24 범위 전환(#12) 기준으로 재정렬했다. 진행은 이슈 → 브랜치 → 구현 → PR 순서로 한다.

## W1 · 기반

| 이슈 | 작업 | 라벨 | 상태 |
|------|------|------|------|
| [#12](https://github.com/UnderCruzer/hancut-inspection/issues/12) | 범위 전환 — 데이터·계획 재작성 | `docs` `data` `ai` | 진행 중 |
| [#16](https://github.com/UnderCruzer/hancut-inspection/issues/16) | 과제 재정의 — Learning to Defer | `ai` `docs` `experiment` | 진행 중 |
| [#14](https://github.com/UnderCruzer/hancut-inspection/issues/14) | PIDray 확보와 무결성·분포 확인 | `data` `chore` | **다음** |
| [#2](https://github.com/UnderCruzer/hancut-inspection/issues/2) | 이용 조건 확인 — 학술 이용·재배포·인용 | `data` `docs` | 열림 |
| [#3](https://github.com/UnderCruzer/hancut-inspection/issues/3) | 1인 작업 방식과 보드 준비 | `chore` `infra` | 열림 |
| [#4](https://github.com/UnderCruzer/hancut-inspection/issues/4) | GPU 자원 확인과 예산 상한 | `infra` | 열림 · 제약 기록됨 |

## W2 · 데이터와 평가 코드

| 이슈 | 작업 | 라벨 | 상태 |
|------|------|------|------|
| [#5](https://github.com/UnderCruzer/hancut-inspection/issues/5) | 라벨 구조 확인 — 박스 단위 vs 이미지 단위 정답 | `data` `ai` | 열림 |
| [#6](https://github.com/UnderCruzer/hancut-inspection/issues/6) | PIDray 라벨 로더 — 인덱스 CSV | `ai` `feature` | 열림 |
| [#7](https://github.com/UnderCruzer/hancut-inspection/issues/7) | ~~SIXray 층화 서브셋~~ | `data` `ai` | **닫힘 — 확보 불가** |
| [#9](https://github.com/UnderCruzer/hancut-inspection/issues/9) | 평가 스크립트 — 예측 CSV에서 E2 표 | `ai` `qa` `feature` | PR #11 |
| [#13](https://github.com/UnderCruzer/hancut-inspection/issues/13) | `facility` → `item` 리네임 | `chore` `ai` `backend` `mobile` | 열림 |

**순서 주의** — #14 → #5 → #6 → #7. #5가 선행 조건이다.
박스 점수를 이미지 한 장의 점수로 어떻게 바꾸느냐(E1)가 여기서 갈리고, 그게 틀리면 E2 이후가 전부 무의미해진다.

## 나중에

| 이슈 | 작업 |
|------|------|
| [#10](https://github.com/UnderCruzer/hancut-inspection/issues/10) | W11 선택 실험 하나 — E5 가림 난이도 vs E6 재검 사유 제시 |

## 닫힌 것

| 이슈 | 이유 |
|------|------|
| [#1](https://github.com/UnderCruzer/hancut-inspection/issues/1) | AI Hub 신청. `us-east-1` 국외 반출 제약으로 취소 (D9). #14가 대체 |
| [#7](https://github.com/UnderCruzer/hancut-inspection/issues/7) | SIXray 서브셋. 비Baidu 미러가 요금 미납으로 정지돼 확보 불가 |
| [#8](https://github.com/UnderCruzer/hancut-inspection/issues/8) | 640px 축소. 보류 — PIDray는 축소 없이 쓴다. #7 용량 실측 후 재판단 |

## 이미 준비된 것

| 모듈 | 내용 | 테스트 |
|------|------|--------|
| `ml/hancut/eval/zones.py` | 판정 3구간, 놓침률 상한 기반 임계값 탐색 (E2·E3·E4) | 23 |
| `ml/hancut/eval/cli.py` | 예측 CSV → 구간 표 + 임계값 JSON | 25 |
| `ml/hancut/data/subset.py` | 층화 추출, 시드 재현, 부족 층 보고 | 11 |
| `ml/configs/items.json` | 위해물품 12종, phase 1은 PIDray∩SIXray 5종 | — |
| `server/` | 판정 API — 모델 미탑재 시 503 | 14 |
| `app/` | Flutter 골격 — 미지의 판정 값은 '확인 필요'로 | 6 |

## 아직 없는 것

- 라벨 로더 (#6) — 스키마 확인 전이라 추측으로 쓰지 않았다
- 검출 모델 연결 — 사전학습 검출기를 쓴다. 직접 만들지 않는다
- **판독관 시뮬레이터** — E3·E4의 입력. 난이도 라벨 범위 확인(#14) 후
- **L2D 손실과 위임 예산 비교 스크립트** — 기존 손실을 가져다 쓴다
- 장비 교차 실험 코드 (E5) — #14에서 장비 메타데이터가 실제로 확인된 뒤
