# 인수인계 — 보안검색 X-ray 전환 (2026-09-24)

## 현재 방향

한컷점검: 보안검색 X-ray 사진을 놓침률 상한 아래에서 **자동통과 / 재검 / 자동적발** 세 구간으로 나눠,
판독관이 직접 봐야 할 가방을 몇 %까지 줄일 수 있는지 정한다.

**새 탐지기를 만들지 않는다.** 사전학습 검출기를 쓰고 운영점 설계가 기여다.
`docs/plan.md`, `docs/data.md`, `docs/decisions.md`의 D9를 먼저 읽는다.

## 환경

AWS **us-east-1**(학교 지정, 변경 불가). 이 제약 때문에 AI Hub 데이터를 쓸 수 없다 — D9 참조.
DLAMI PyTorch 2.13 / CUDA 13.0, A10G 23028MiB, GPU True 확인. 디스크 192GB.
`source /opt/pytorch/bin/activate` 로 활성화한다. 맥/Windows는 원격 조작용이다.

IAM 정책 `ControlOnlyOwnResources` — 본인이 생성한 리소스만 제어할 수 있다.
**보안 그룹은 인스턴스 생성 시점에 지정해야 한다.** 나중에 바꾸려면 권한 오류가 난다.
기존 `launch-wizard-*` 를 재사용하거나 생성 시 함께 만든다.

GitHub: https://github.com/UnderCruzer/hancut-inspection
작업 전 최신 상태를 확인하고 pull 한다.

## 다음 작업

1. **#14** — PIDray 확보. https://github.com/bywang2018/security-dataset
   장수·용량·class_id 실제 대응·**장비 3종 식별자와 장비별 장수**·split 누수를 확인한다.
   `ml/configs/items.json` 의 `class_id` 와 `devices.ids` 를 여기서 채운다.
2. **#5** — 박스 단위 라벨에서 이미지 한 장의 정답을 어떻게 정의할지 정한다. E1의 갈림길이다.
3. **#6** — 라벨 로더. #5 전에는 쓰지 않는다 (D5).
4. 사전학습 검출기 연결 → 박스 점수 추출 → 기존 #9 평가 CLI(`ml/scripts/evaluate.py`)에 물린다.

## 주의

PIDray·SIXray는 **학술 목적 한정, 재배포 금지**다. 원본을 저장소에 올리지 않는다.
AI Hub 데이터는 미국 AWS에 올리지 않는다.
데이터·가중치·비밀키는 Git에 넣지 않는다.
모델은 아직 학습되지 않았고 기존 테스트는 기반 코드 테스트다.
