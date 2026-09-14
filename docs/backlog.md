# W1–W2 백로그

이슈로 옮길 작업 목록. 이슈 → 브랜치 → 구현 → PR 순서로 진행한다.

## W1 · 기반

- [ ] AI Hub 가입·데이터 신청, 샘플 데이터 다운로드 — `docs`
- [ ] 이용약관 확인 (재배포·클라우드 업로드) — `docs`
- [ ] 팀 역할 확정, Jira 보드·Slack 채널 개설 — `chore`
- [ ] GPU 자원 확인 (학교 실습실 / Colab) — `infra`

## W2 · 데이터와 평가 코드

- [ ] 샘플로 라벨 구조 확인 (`docs/data.md` W2 체크리스트) — `ml`
- [ ] 라벨 JSON 파서 → 인덱스 CSV (`image_id, facility, compliant, source, reasons`) — `ml`
- [ ] 8종 층화 서브셋 추출 실행, 서브셋 목록 커밋 — `ml`
- [ ] 640px 축소 실행, 전후 용량 기록 — `ml`
- [ ] 평가 스크립트: 모델 예측 CSV → `zones.sweep()` 결과 표 — `ml`

## 이미 준비된 것

- `ml/hancut/eval/zones.py` — 판정 3구간, 놓침률 상한 기반 임계값 탐색 (E2)
- `ml/hancut/data/subset.py` — 시설·준수 여부별 층화 추출
- `ml/hancut/data/resize.py` — 640px 축소, 박스 좌표 변환
- `server/` — 판정 API 골격 (모델 미탑재 시 503)
- `app/` — Flutter 앱 골격
