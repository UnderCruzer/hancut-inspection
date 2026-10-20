#!/usr/bin/env python3
"""설정 F 를 시험 분할 seed 여러 개로 반복 평가한다 (#45).

    python3 ml/scripts/f_seeds.py --arm E1=ml/runs/dense_20261019/e1/dense --arm E1+960=ml/runs/fusion_x \\
        --cap 0.05 --out ml/runs/f_seeds_20261020

seed 하나의 F 는 운에 좌우된다(#43: 같은 모델의 F·5% 자동 처리가 seed 에 따라 23.8~44.9%).
그래서 seed 마다 시험 사진을 다시 절반으로 나눠 보정 · 평가하고, 평균 · 범위와 상한을 넘은 횟수를 낸다.

δ 기본값은 0.05/36 — 품목 12 × 난이도 3 = 36칸을 **동시에** 95% 로 보장한다 (#43).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.eval import cli, resplit  # noqa: E402
from hancut.eval.grouped import DIFFICULTIES, grouped_table, worst_cell  # noqa: E402

CELLS = 36


def parse_arms(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"'{spec}' 를 읽지 못했다. 형식은 이름=예측 폴더")
        out[name] = Path(path)
    return out


def run_seed(test: list[dict], cap: float, delta: float, seed: int, fraction: float = 0.5) -> dict:
    calib_ids, _ = resplit.split_images(resplit.image_table(test), fraction, seed)
    test_calib, test_eval = resplit.partition(test, calib_ids)
    table = grouped_table(test_calib, cap, "upper", delta)
    total = cli.point_for(test_eval, table)
    items = sorted({r["item"] for r in test_eval})
    cell, cell_miss, cell_n = worst_cell(test_eval, table, items)
    return {"seed": seed, "auto": total["auto_rate"], "miss": total["miss_rate"], "false_alarm": total["false_alarm_rate"],
            "worst_cell": cell, "worst_miss": cell_miss, "worst_n": cell_n, "over": cell_miss > cap}


def summarize(runs: list[dict]) -> dict:
    def stat(key):
        vals = [r[key] for r in runs]
        return {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    return {"n": len(runs), "auto": stat("auto"), "miss": stat("miss"), "false_alarm": stat("false_alarm"),
            "over": sum(r["over"] for r in runs), "worst_miss_max": max(r["worst_miss"] for r in runs)}


def table_lines(summary: dict[str, dict]) -> list[str]:
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    out = ["| 갈래 | 자동 처리 평균 (범위) | 실측 놓침 (범위) | 오경보 (범위) | 상한을 넘은 횟수 | 가장 나쁜 칸 최대 |",
           "|---|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        a, m, f = s["auto"], s["miss"], s["false_alarm"]
        out.append(f"| {name} | {pct(a['mean'])} ({pct(a['min'])}~{pct(a['max'])}) | {pct(m['min'])}~{pct(m['max'])} | "
                   f"{pct(f['min'])}~{pct(f['max'])} | {s['over']}/{s['n']} | {pct(s['worst_miss_max'])} |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="이름=예측 폴더 (여러 번)")
    parser.add_argument("--cap", type=float, nargs="+", default=[0.05])
    parser.add_argument("--delta", type=float, default=0.05 / CELLS, help="기본 0.05/36 — 36칸 동시 95%%")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    arms = parse_arms(args.arm)
    for name, folder in arms.items():
        missing = [f"test_{d}.csv" for d in DIFFICULTIES if not (folder / f"test_{d}.csv").is_file()]
        if missing:
            raise SystemExit(f"{name}: {folder} 에 없다 {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# 설정 F — seed 반복 (#45)", "", f"seed {args.seeds[0]}~{args.seeds[-1]} ({len(args.seeds)}회) · "
             f"시험 사진 절반 보정 · 절반 평가 · δ = {args.delta:.5f}", ""]
    record = {}
    for cap in args.cap:
        summary = {}
        for name, folder in arms.items():
            test = [r for d in DIFFICULTIES for r in cli.read_predictions(folder / f"test_{d}.csv")]
            start = time.time()
            runs = [run_seed(test, cap, args.delta, s) for s in args.seeds]
            summary[name] = summarize(runs)
            record[f"{name}@{cap}"] = {"summary": summary[name], "runs": runs}
            print(f"{name} · 상한 {cap:.0%}: 자동 {summary[name]['auto']['mean']:.1%}, 넘음 {summary[name]['over']}/{len(runs)} "
                  f"({time.time() - start:.0f}초)")
        lines += [f"## 상한 {cap:.0%}", ""] + table_lines(summary) + [""]
    (args.out / "runs.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "\n".join(lines)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
