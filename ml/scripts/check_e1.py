#!/usr/bin/env python3
"""E1 결과를 해석하기 전에 파이프라인 버그부터 확인한다 (#23).

    python3 ml/scripts/check_e1.py

1. 네 어노테이션 파일의 품목 번호표가 같은가
   YOLO 라벨은 번호로 품목을 잇는다. 파일마다 번호표가 다르면 정답이 뒤섞인다.
2. 예측 CSV 의 정답(y_true)이 인덱스(data/index.csv)의 정답과 같은가
   인덱스는 번호가 아니라 이름으로 만들었다. 두 경로가 같은 답을 내야 한다.
3. 난이도별 놓침률 — 같은 임계값이 easy / hard / hidden 어디서 무너지는가
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN = next((p for p in (REPO / "data/pidray/pidray/annotations", REPO / "data/pidray/annotations")
            if (p / "xray_train.json").is_file()), None)
INDEX = REPO / "data" / "index.csv"
PRED = REPO / "ml" / "runs" / "predictions"
RUNS = REPO / "ml" / "runs"


def check_categories() -> bool:
    print("## 1. 품목 번호표\n")
    if ANN is None:
        print("어노테이션 폴더를 찾지 못했다.")
        return False
    ref, ok = None, True
    for f in ("xray_train", "xray_test_easy", "xray_test_hard", "xray_test_hidden"):
        table = {c["id"]: c["name"] for c in json.loads((ANN / f"{f}.json").read_text())["categories"]}
        ref = ref or table
        same = table == ref
        ok &= same
        print(f"- {f:<18} {'같음' if same else '**다름** ' + str(table)}")
    return ok


def check_truth() -> bool:
    print("\n## 2. 정답 대조 (예측 CSV vs 이름으로 만든 인덱스)\n")
    test_all = PRED / "test_all.csv"
    if not INDEX.is_file():
        print("data/index.csv 가 없다. `python3 ml/scripts/build_index.py` 를 먼저 돌린다.")
        return False
    if not test_all.is_file():
        print(f"{test_all} 가 없다.")
        return False
    index = {}
    with INDEX.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "test":
                index[(r["image_id"], r["item"])] = r["y_true"]
    bad, n = Counter(), 0
    with test_all.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            if index.get((r["image_id"], r["item"])) != r["y_true"]:
                bad[re.sub(r"\d+", "", r["image_id"])] += 1
    total = sum(bad.values())
    print(f"{n:,}행 중 불일치 **{total:,}**" + (f" — {dict(bad)}" if bad else ""))
    if n != len(index):
        print(f"행 수도 다르다: 예측 {n:,} vs 인덱스 {len(index):,}")
    return total == 0 and n == len(index)


def per_difficulty() -> None:
    print("\n## 3. 난이도별 시험셋 놓침률 (보정셋에서 정한 같은 임계값)\n")
    metrics = {}
    for d in ("easy", "hard", "hidden"):
        path = RUNS / f"e2_{d}" / "metrics.json"
        if not path.is_file():
            print(f"{path} 가 없다. 4번 명령(난이도별 평가)을 먼저 돌린다.")
            return
        metrics[d] = json.loads(path.read_text())["test"]
    print(f"{'품목':<11}{'easy':>9}{'hard':>9}{'hidden':>9}   (양성 easy/hard/hidden)")
    for item in sorted(metrics["easy"], key=lambda k: (k != "전체", k)):
        cells = "".join(f"{(metrics[d][item]['miss_rate'] or 0):>9.1%}" for d in metrics)
        counts = "/".join(str(metrics[d][item]["n_threat"]) for d in metrics)
        print(f"{item:<11}{cells}   ({counts})")


def main() -> int:
    cats = check_categories()
    truth = check_truth()
    per_difficulty()
    print("\n## 판정\n")
    if cats and truth:
        print("라벨 경로에는 버그가 없다. 시험셋 놓침률은 파이프라인 문제가 아니라 실제 결과다.")
        return 0
    print("**라벨이 어긋났다.** 위 결과는 해석하지 말고 변환부터 고친다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
