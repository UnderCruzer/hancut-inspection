# 작업 방식

혼자 하는 프로젝트지만 이슈와 PR을 거친다. 리뷰어가 없어도 **왜 그렇게 했는지가 기록으로 남기 때문**이다. 이 기록이 그대로 포트폴리오가 되고, 면접에서 "본인이 한 게 뭡니까"에 대한 답이 된다.

## 흐름

```
이슈 생성 → 브랜치 → 커밋 → PR → 머지 → 이슈 닫기
```

1. **이슈를 먼저 만든다.** 코드를 건드리기 전에 목표와 완료 조건을 적는다. 라벨은 항상 붙인다.
2. **브랜치를 판다.** `main`에 직접 커밋하지 않는다.
3. **커밋은 작업 단위로 나눈다.** 여러 파일을 몰아서 한 번에 찍지 않는다.
4. **PR을 연다.** 템플릿의 체크리스트를 채운다. 혼자여도 변경 이유를 적는다.
5. **CI가 초록일 때만 머지한다.**

## 브랜치 이름

| 접두어 | 용도 | 예 |
|--------|------|-----|
| `feat/` | 새 기능 | `feat/issue-6-label-parser` |
| `fix/` | 버그 수정 | `fix/issue-21-zone-boundary` |
| `exp/` | 실험 수행 | `exp/issue-12-e1-two-stage` |
| `docs/` | 문서만 | `docs/issue-2-data-terms` |
| `chore/` | 설정·정리 | `chore/issue-3-project-board` |

이슈 번호를 반드시 넣는다.

## 커밋 메시지

형식은 `type(scope): 영어 설명`. 저장소 이력이 영어로 되어 있어 맞춘다. 이슈·PR 본문은 한국어로 쓴다.

```
feat(ml): add stratified subset sampler with reproducible seed
fix(server): reject uploads over the size limit before reading the body
exp(ml): record E1 two-stage comparison on 8 facility types
docs: note the W2 label schema findings
```

type: `feat` `fix` `exp` `refactor` `test` `docs` `ci` `chore`
scope: `ml` `server` `app` 또는 생략

**`Co-Authored-By` 줄은 넣지 않는다.**

## 라벨

| 라벨 | 뜻 |
|------|-----|
| `feature` `bug` `chore` `docs` | 작업 성격 |
| `ai` `data` `backend` `mobile` `infra` | 영역 |
| `experiment` | E1–E7 실험 |
| `qa` `security` | 테스트·안전 |

이슈마다 2~3개를 조합한다. 담당자는 라벨이 아니라 assignee로 지정한다.

## PR을 열기 전에

```bash
cd ml     && ../.venv/bin/python -m pytest -q
cd server && ../.venv/bin/python -m pytest -q
cd app    && flutter analyze && flutter test
```

세 개가 모두 통과해야 한다. CI에서 같은 것을 다시 돌린다.

## 하지 말 것

- 이미지·라벨 JSON·모델 가중치 커밋 (AI Hub 재배포 제한)
- 테스트 없이 새 모듈 추가
- 판정 임계값을 코드에 하드코딩 — 학습 쪽 E2 결과를 파일로 받는다
- 모델이 없을 때 그럴듯한 가짜 결과 반환
