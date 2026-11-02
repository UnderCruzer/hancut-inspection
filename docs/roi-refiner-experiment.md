# 물체 영역 재분류 실험 · 2026-10-27

**결과: calib 에서 기각. 시험셋은 평가하지 않았다.** 수치는 [실험 결과](results.md)의 #48 항목을 참고한다.

## 목적

낮은 라이터 검출 AP와 칼·가위 혼동을 줄일 수 있는지 확인한다. 기존 반복학습·해상도 증가·타일·MixUp
결과를 그대로 반복하지 않고, E1이 찾은 물체 영역을 별도 분류기로 다시 구분한다.
이 실험에서 ROI 분류 정확도가 높아지는 것만으로 검출 성능 개선을 선언하지 않는다.

## 고정 설계

- E1 가중치는 수정하지 않는다. 12개 물체+배경의 MobileNet V3 Small 분류기만 학습한다.
- ImageNet 사전학습 가중치를 사용하며, 학습 사진 25,039장에서 정답 물체 crop과 배경 crop을 만든다.
  배경은 검출기의 오탐 후보를 우선하고, 없다면 seed 0으로 무작위 영역을 선택한다.
  영역에 실제 물체가 포함되지 않도록 모든 정답 박스의 가려진 면적 비율을 검사한다.
- crop은 문맥 20%를 포함하고 종횡비를 유지해 224×224로 패딩한다. 학습 때 뒤집기·작은 회전·색 변화·
  일부 회색조 변환을 사용한다. 첫 에폭은 분류 헤드만, 이후 에폭은 전체 보조 모델을 학습한다.
- 8에폭, AdamW, lr 0.0003, batch 128, seed 0. 검증 사진 1,474장의 물체 클래스 평균 재현율로
  체크포인트를 선택한다. 모델 학습에 calib나 시험 이미지를 사용하지 않는다.
- 실제 추론에서 정답 crop을 주지 않는다. E1 검출 결과 중 점수 0.05 이상 상위 20개 후보만 재분류한다.
- 라이터와 칼만 점수를 조합한다. 나머지 10개 품목의 검출 결과는 그대로 유지되는지 매 이미지 검사한다.

## 후보 선택 및 평가

calib 2,944장만으로 품목별 조합 비율 0, 0.25, 0.5, 0.75, 1을 비교한다.
기존 클래스 점수와 `후보 검출 점수 × 보조 분류기의 해당 클래스 점수`를 조합하고 품목별 NMS를 적용한다.
비율 0은 기존 검출기 결과를 정확히 유지한다.

채택 후보는 calib에서 모두 만족해야 한다.

- AP50이 0.005 이상 증가
- AP50–95 하락 0.005 이내
- conf 0.25 / IoU 0.5의 정밀도 하락 2%p 이내, 재현율 하락 1%p 이내

조건을 충족하는 후보 중 AP50이 가장 높은 비율을 고른다. 두 품목 모두 비율 0이면 기각하고 시험셋을
추가 평가하지 않는다. 후보가 남으면 설정과 모델 SHA256을 파일로 고정한 후 easy/hard/hidden을 평가한다.
주요 보고 항목은 품목별 AP·정밀도·재현율 및 분모, 이미지×품목의 놓침·재검·오경보·자동 처리율,
추가 crop 수와 실행 시간이다. 현재 dense API와의 비교 없이 API를 교체하지 않는다.

PIDray 시험셋은 앞선 실험에 여러 번 사용한 탐색 데이터다. 이 결과를 독립 검증이나 운영 보장으로
해석하지 않으며, 시험셋 결과를 보고 조합 비율이나 체크포인트를 다시 고르지 않는다.

## 실행

설정: `ml/configs/roi_refiner_20261027.json`.
코드: `ml/scripts/run_roi_refiner.py`, `ml/hancut/eval/roi.py`.

```bash
python ml/scripts/run_roi_refiner.py \
  --weights /home/ubuntu/hancut-inspection/ml/runs/detect/e1_smoke/weights/best.pt \
  --out ml/runs/roi_full
```

AWS 별도 작업 폴더: `/home/ubuntu/hancut-roi-20261027`.
진행 확인: `tmux attach -t hancut-roi` 또는 `cat ml/runs/roi_full/status.json`.
소규모 실행은 `--smoke`와 새 출력 폴더를 사용한다. 이 모드의 성능 수치는 채택 근거로 사용하지 않는다.
원본 이미지·crop·체크포인트는 `ml/runs/`에만 저장하며 커밋하지 않는다.

구현 참고: [Torchvision MobileNet V3 Small](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.mobilenet_v3_small.html),
[Ultralytics 검출 지표](https://docs.ultralytics.com/reference/utils/metrics/).
