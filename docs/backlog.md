# W1–W2 백로그

이슈로 옮긴 상태다. 진행은 이슈 → 브랜치 → 구현 → PR 순서로 한다.

## W1 · 기반

| 이슈 | 작업 | 라벨 |
|------|------|------|
| [#1](https://github.com/UnderCruzer/hancut-inspection/issues/1) | AI Hub 데이터 신청과 샘플 다운로드 | `data` `chore` |
| [#2](https://github.com/UnderCruzer/hancut-inspection/issues/2) | 데이터 이용약관 확인 — 재배포·클라우드 조건 | `data` `docs` |
| [#3](https://github.com/UnderCruzer/hancut-inspection/issues/3) | 1인 작업 방식과 보드 준비 | `chore` `infra` |
| [#4](https://github.com/UnderCruzer/hancut-inspection/issues/4) | GPU 자원 확인과 예산 상한 합의 | `infra` |

## W2 · 데이터와 평가 코드

| 이슈 | 작업 | 라벨 |
|------|------|------|
| [#5](https://github.com/UnderCruzer/hancut-inspection/issues/5) | 샘플 데이터로 라벨 구조 확인 | `data` `ai` |
| [#6](https://github.com/UnderCruzer/hancut-inspection/issues/6) | 라벨 JSON 파서 — 인덱스 CSV 생성 | `ai` `feature` |
| [#7](https://github.com/UnderCruzer/hancut-inspection/issues/7) | 8종 층화 서브셋 추출 실행 | `data` `ai` |
| [#8](https://github.com/UnderCruzer/hancut-inspection/issues/8) | 640px 축소 실행과 용량 실측 | `data` `ai` |
| [#9](https://github.com/UnderCruzer/hancut-inspection/issues/9) | 평가 스크립트 — 예측 CSV에서 E2 표 생성 | `ai` `qa` `feature` |

**순서 주의** — #5가 선행 조건이다. 준수/미준수가 박스 단위인지 이미지 단위인지에 따라 #6의 파서 설계와 E1의 모델 구조가 갈린다. #6 → #7 → #8 순으로 이어진다. #9는 데이터와 무관하게 먼저 진행할 수 있다.

## 이미 준비된 것

| 모듈 | 내용 | 테스트 |
|------|------|--------|
| `ml/hancut/eval/zones.py` | 판정 3구간, 놓침률 상한 기반 임계값 탐색 (E2) | 23 |
| `ml/hancut/data/subset.py` | 시설 × 준수 여부 층화 추출, 시드 재현, 부족 층 보고 | 11 |
| `ml/hancut/data/resize.py` | 640px 축소, 박스 좌표 변환, 전후 용량 리포트 | 10 |
| `server/` | 판정 API — 모델 미탑재 시 503, 시설별 임계값 | 14 |
| `app/` | Flutter 골격 — 판정 구간 규약 공유, 미지의 값은 '확인 필요' | 6 |

## 나중에 결정할 것

| 이슈 | 작업 |
|------|------|
| [#10](https://github.com/UnderCruzer/hancut-inspection/issues/10) | W13 선택 실험 하나 고르기 — E6 연한 OCR vs E5 점검표 초안 |

## 아직 없는 것

- 라벨 파서 (#6) — 스키마 확인 전이라 추측으로 쓰지 않았다
- 검출·판정 모델 — W3 이후
- 촬영 화면과 정정 흐름 — W9–W10
- OCR 연한 판정 **또는** 점검표 생성 — W13에 하나만 (#10)
