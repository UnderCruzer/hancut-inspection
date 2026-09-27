# Windows 인수인계 — 2026-09-24 업데이트

## 최신 결정 (이전 소방시설 계획보다 우선)

사용자가 VisA PCB 결함 검사로 전환 승인. 서비스명 한컷검수.
학교가 AWS 버지니아 사용을 지정했으나 AI Hub 국외 반출 합의는 없어 데이터 교체.
VisA는 CC BY 4.0 공개 데이터이며 기존 AI Hub 합의 절차와 별개다.

현재 계획은 docs/plan.md, 다운로드는 docs/data.md를 읽는다.
GitHub: https://github.com/UnderCruzer/hancut-inspection
작업 브랜치: feat/issue-9-evaluation-cli / PR #11 (최신 원격 상태 확인).
저장소 이름은 그대로 유지한다. git pull 전 로컬 변경을 확인한다.

## 확인된 환경과 구현

AWS A10G 23028MiB, PyTorch2.13.0+cu130, GPU True. 디스크 여유180G는 과거 실측.
source /opt/pytorch/bin/activate로 활성화. 맥/Windows는 원격 조작용이다.
평가 CLI와 서버·앱 골격 존재, 로컬86개 테스트 통과 이력. 실제 모델 학습/VisA 로더는 아직 없음.
소방시설 설정/화면/문자열이 코드에 남아 있으므로 VisA용으로 순차 전환해야 한다.

## 바로 다음 작업

1. AWS에서 공식 VisA TAR(약1.93GB) 다운로드, 크기/내부 구조 확인.
2. pcb1 라벨·마스크·공식 split 확인, 검증/시험 분리 방안 확정.
3. 데이터 로더 및 기준선 구현, 소규모 실행 후 비용·메모리 실측.
4. 점수 보정 → 기존 E2 CLI 연결 → 정상/확인/결함 서비스 통합.

원시 이상 점수를 확률로 간주하지 않는다. 시험셋으로 임계값/보정기를 맞추지 않는다.
공식 벤치마크 분할 변경 시 자체 프로토콜로 명시한다.
데이터·모델·키는 커밋하지 않는다. 이전 맥 데이터는 AI Hub 샘플/검증 라벨이며 VisA는 아직 없다.

## 새 채팅에 전달

이 문서와 AGENTS.md, docs/plan.md, docs/data.md를 읽고 VisA PCB 검수 AX 프로젝트를 이어서 진행해줘.
소방시설 데이터 다운로드 안내는 과거 계획이므로 재개하지 마. 우선 실제 VisA 다운로드 여부와 파일 구조를 확인하고 로더부터 구현해줘.

이전 상세 이력은 archive-aihub/windows-handoff.md에 보관한다.
