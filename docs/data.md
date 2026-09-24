# 현재 데이터 — Real-IAD 1024

2026-09-24 사용자 승인: 주 데이터는 Real-IAD의 realiad_1024 버전.
AI Hub는 현 실험에서 사용하지 않으며 VisA는 보조 비교용으로 보관한다.

## 선택과 용량

공식 안내: 1024×1024 버전 약53GB, 고해상도 raw 약507GB.
53GB는 제공기관의 근사 안내다. 실제 파일별 합계 및 압축 해제 후 크기는 다운로드 전후 별도 확인한다.
Hugging Face 저장소 전체 다운로드는 여러 버전까지 받을 수 있으므로 사용하지 않는다.

2026-09-24 공식 파일 목록 확인:

| realiad_1024 내부 파일 | 표시 크기 |
|---|---:|
| pcb.zip | 1.50GB |
| audiojack.zip | 1.06GB |
| usb.zip | 1.31GB |
| switch.zip | 1.41GB |

PCB를 먼저 받아 로더를 검증하고, 위 4종(약5.28GB + 분할 JSON)을 초기 실험 후보로 삼는다.
전체 데이터 수량을 이 4종 수량으로 오인하지 않는다. 제품별 정상/결함 및 독립 제품 수는 라벨 확보 후 집계해야 한다.
realiad_jsons.zip의 분할 정보도 필요하다. 제품별 ZIP 내부 마스크·경로는 아직 읽지 않았다.

## 이용 및 다운로드 순서

1. https://huggingface.co/datasets/Real-IAD/Real-IAD 에 로그인.
2. 연구 목적 및 연락처 공유/이용 조건을 읽고 본인이 동의하여 접근 요청. 공식 안내는 자동 승인이나 실제 계정 상태는 미확인.
3. AWS에서 접근 가능한 계정으로 인증 후 realiad_1024/pcb.zip과 분할 JSON만 선택 다운로드.
4. 여유 디스크 확인, ZIP 해제 용량/마스크/분할 확인 후 범위 확대.

공식 데이터 카드는 CC BY-NC-SA 4.0 및 연구 목적 전용을 명시한다. 출처·라이선스·변경 사항을 기록한다.
원본/모델/인증 토큰은 Git에 넣지 않는다. 원시 고해상도 전체를 다운로드하지 않는다.

## 평가 시 주의

동일 제품의 5개 시점이 학습/검증/시험에 흩어지지 않도록 제품 ID로 묶는다.
사진 수와 독립 제품 수를 따로 집계한다. 1024 파일 사용은 모델 입력도 반드시1024여야 한다는 뜻은 아니다.
미세 결함 손실을 피하기 위해 마스크로 실제 크기를 보고 모델 입력 크기/타일 사용을 결정한다.
학습·검증·시험 역할과 점수 보정은 실험 프로토콜에 기록하고 시험 데이터로 임계값을 선택하지 않는다.

## 출처

- https://realiad4ad.github.io/Real-IAD/
- https://huggingface.co/datasets/Real-IAD/Real-IAD
- https://huggingface.co/datasets/Real-IAD/Real-IAD/tree/main/realiad_1024
- Wang et al., Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection, CVPR 2024.
