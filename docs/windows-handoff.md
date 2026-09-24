# Windows 인수인계 — 소방시설 주제 복구 (2026-09-24)

## 현재 방향

한컷점검: 소화기 사진의 압력계 상태·부식 의심 위치를 찾고 담당자 검토와 점검 기록을 돕는다.
사용자가 소방시설 주제를 유지하라고 정정했다. VisA/Real-IAD 제품 검사 전환은 취소됐다.
계획 docs/plan.md, 데이터 docs/data.md, 확인 근거 docs/fire-dataset-review.md를 먼저 읽는다.

## 환경

AWS us-east-1(학교 지정), A10G 23028MiB / PyTorch 2.13.0+cu130 / GPU True 확인 이력.
/opt/pytorch/bin/activate 활성화. 맥/Windows는 원격 조작용이다. 디스크 여유180G는 과거 측정이므로 다시 확인한다.
GitHub: https://github.com/UnderCruzer/hancut-inspection
브랜치 feat/issue-9-evaluation-cli / PR #11 (작업 전 최신 상태 확인). 로컬 변경 확인 후 pull.

## 다음 작업

1. Weqaa v18 공개 다운로드 페이지에서 로그인 후 이미지+라벨 확보. 계정 생성/약관 동의는 사용자가 수행한다.
2. 실제 export의 클래스/라벨 형식/라이선스/용량/원본 그룹 수 확인. 로그인 전 공개 UI에서 gauge_bad polygon 한 건만 직접 확인했다.
3. 같은 영상의 프레임이 분할을 넘는 문제를 해결하고 기준선 로더·학습 구현.
4. 상태별 검출·점수 보정 후 기존 #9 평가 CLI를 연결한다.

AI Hub 데이터는 학교에 국외 반출 합의가 없어 미국 AWS에 올리지 않는다. VisA/Real-IAD 추가 다운로드는 현재 계획이 아니다.
데이터·가중치·비밀키는 Git에 넣지 않는다. 모델은 아직 학습되지 않았으며 기존 테스트는 기반 코드 테스트다.
