#!/usr/bin/env python3
"""E2 보정 방식 네 가지를 같은 평가 사진에서 비교한다 (#25).

    python3 ml/scripts/e2_compare.py

| | 보정 사진 | 상한 판정 |
|---|---|---|
| A | 학습 분포 (calib.csv) | 표본 비율 — 첫 E2 방식 |
| B | 학습 분포 | 95% 신뢰 상한 |
| C | 시험 분포 (시험셋 절반) | 표본 비율 |
| D | 시험 분포 | 95% 신뢰 상한 |
| E | 시험 분포 | 표본 비율, 난이도 셋 모두에서 동시에 (#27) |
| F | 시험 분포 | 95% 신뢰 상한, 난이도 셋 모두에서 동시에 (#27) |

E·F 도 품목마다 기준은 하나다. 그 하나가 easy·hard·hidden 모두에서 상한을 지키도록 고른다.
검색대에서는 숨긴 가방인지 미리 알 수 없으므로, 난이도별로 기준을 따로 쓰는 방식은 비교하지 않는다.

평가는 모든 설정이 시험셋의 같은 나머지 절반으로 한다. 검출기를 다시 돌리지 않는다.
결과는 ml/runs/e2_compare/<시각>/ 에 남고, 요약은 summary.md 로도 저장된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.eval import cli, resplit, zones  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIFFICULTIES = ("easy", "hard", "hidden")
# (키, 보정 이름, 판정 이름, 보정 사진, 판정 방식, 난이도 동시 보장)
CONFIGS = [
    ("A", "학습", "비율", "train", "empirical", False),
    ("B", "학습", "95%상한", "train", "upper", False),
    ("C", "시험", "비율", "test", "empirical", False),
    ("D", "시험", "95%상한", "test", "upper", False),
    ("E", "시험", "비율·난이도", "test", "empirical", True),
    ("F", "시험", "95%상한·난이도", "test", "upper", True),
]
FALSE_ALARM_CAP = 0.05


def grouped_table(rows, cap: float, bound: str, delta: float) -> dict:
    """품목마다 기준 하나. easy·hard·hidden 모두에서 동시에 상한을 지키도록 고른다 (#27)."""
    def groups(subset):
        out = {}
        for r in subset:
            y, s = out.setdefault(resplit.difficulty_of(r["image_id"]), ([], []))
            y.append(r["y_true"])
            s.append(r["score"])
        return out

    fit = dict(bound=bound, delta=delta)
    table = {}
    for item in sorted({r["item"] for r in rows}):
        t = zones.fit_thresholds_grouped(groups([r for r in rows if r["item"] == item]), cap, FALSE_ALARM_CAP, **fit)
        table[item] = {"low": t.low, "high": t.high, "miss_rate_cap": cap}
    t = zones.fit_thresholds_grouped(groups(rows), cap, FALSE_ALARM_CAP, **fit)
    table["default"] = {"low": t.low, "high": t.high, "miss_rate_cap": cap}
    return table


def worst_cell(rows, table, items) -> tuple[str, float, int]:
    """품목 × 난이도 칸 가운데 놓침률이 가장 높은 칸. E5 의 판정은 모든 칸이 상한 이하인가다."""
    worst = ("", -1.0, 0)
    for item in items:
        for d in DIFFICULTIES:
            cell = [r for r in rows if r["item"] == item and resplit.difficulty_of(r["image_id"]) == d]
            point = cli.point_for(cell, table) if cell else None
            if point and point["miss_rate"] is not None and point["miss_rate"] > worst[1]:
                worst = (f"{item}·{d}", point["miss_rate"], point["n_threat"])
    return worst


def write_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cli.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cli.FIELDS})


def pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, default=REPO / "ml" / "runs" / "predictions")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cap", type=float, default=0.01, help="임계값을 고를 놓침률 상한")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--fraction", type=float, default=0.5, help="시험셋 중 보정에 쓸 비율")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.out_dir or REPO / "ml" / "runs" / "e2_compare" / time.strftime("%Y%m%d-%H%M%S")
    need = ["calib.csv"] + [f"test_{d}.csv" for d in DIFFICULTIES]
    missing = [n for n in need if not (args.pred_dir / n).is_file()]
    if missing:
        print(f"{args.pred_dir} 에 없다: {missing}. predict_scores.py 를 먼저 돌린다.")
        return 2

    train_calib = cli.read_predictions(args.pred_dir / "calib.csv")
    test = [r for d in DIFFICULTIES for r in cli.read_predictions(args.pred_dir / f"test_{d}.csv")]
    calib_ids, _ = resplit.split_images(resplit.image_table(test), args.fraction, args.seed)
    test_calib, test_eval = resplit.partition(test, calib_ids)

    data = out / "data"
    paths = {"train": data / "train_calib.csv", "test": data / "test_calib.csv"}
    write_rows(train_calib, paths["train"])
    write_rows(test_calib, paths["test"])
    write_rows(test_eval, data / "test_eval.csv")

    by_diff = {d: [r for r in test_eval if resplit.difficulty_of(r["image_id"]) == d] for d in DIFFICULTIES}
    items = sorted({r["item"] for r in test_eval})
    n_img = lambda rows: len({r["image_id"] for r in rows})  # noqa: E731
    lines = [
        f"# E2 보정 방식 비교 — 놓침률 상한 {args.cap:.0%}",
        "",
        f"평가 사진: 시험셋 {n_img(test_eval):,}장 (easy {n_img(by_diff['easy']):,} · hard {n_img(by_diff['hard']):,} · "
        f"hidden {n_img(by_diff['hidden']):,}). 모든 설정을 같은 사진으로 평가한다.",
        f"보정 사진: 학습 분포 {n_img(train_calib):,}장 / 시험 분포 {n_img(test_calib):,}장. 95%상한은 신뢰수준 {1 - args.delta:.0%}.",
        "",
        "| 설정 | 보정 | 판정 | 놓침 | 재검 | 자동 | 오경보 | easy 놓침 | hard 놓침 | hidden 놓침 | 가장 나쁜 칸 (품목·난이도) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    per_item = {}
    for key, cal_name, bound_name, source, bound, grouped in CONFIGS:
        run_dir = out / key
        try:
            if grouped:
                table = grouped_table(test_calib, args.cap, bound, args.delta)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "thresholds.json").write_text(
                    json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                argv = [str(paths[source]), "--output-dir", str(run_dir), "--test-csv", str(data / "test_eval.csv"),
                        "--select-cap", str(args.cap), "--miss-caps", str(args.cap),
                        "--bound", bound, "--delta", str(args.delta)]
                cli.run(cli.build_parser().parse_args(argv))
                table = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        except ValueError as exc:
            lines.append(f"| {key} | {cal_name} | {bound_name} | 실패: {exc} | | | | | | | |")
            continue
        total = cli.point_for(test_eval, table)
        diffs = {d: cli.point_for(rows, table) for d, rows in by_diff.items()}
        per_item[key] = {i: cli.point_for([r for r in test_eval if r["item"] == i], table) for i in items}
        cell, cell_miss, cell_n = worst_cell(test_eval, table, items)
        flag = "" if cell_miss <= args.cap else " ❌"
        lines.append(
            f"| {key} | {cal_name} | {bound_name} | **{pct(total['miss_rate'])}** | {pct(total['review_rate'])} | "
            f"{pct(total['auto_rate'])} | {pct(total['false_alarm_rate'])} | "
            + " | ".join(pct(diffs[d]["miss_rate"]) for d in DIFFICULTIES)
            + f" | {cell} {pct(cell_miss)} (n={cell_n}){flag} |"
        )

    if per_item:
        keys = list(per_item)
        # 품목이 있는데 검출기가 그 품목 박스를 하나도 내지 않은 비율. 점수가 0 이면 '없음'과 구분할 수
        # 없으므로, 이 비율이 놓침률 상한보다 크면 자동 통과를 거의 열 수 없다.
        blind = {}
        for i in items:
            pos = [r for r in test_eval if r["item"] == i and r["y_true"] == 1]
            blind[i] = sum(r["score"] == 0.0 for r in pos) / len(pos) if pos else None
        lines += ["", "## 품목별 놓침 / 재검", "",
                  "박스 없음 = 품목이 있는데 검출기가 그 품목 박스를 하나도 내지 않은 비율 (설정과 무관).", "",
                  "| 품목 | 박스 없음 | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
        for i in items:
            cells = [f"{pct(per_item[k][i]['miss_rate'])} / {pct(per_item[k][i]['review_rate'])}" for k in keys]
            lines.append(f"| {i} | {pct(blind[i])} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
