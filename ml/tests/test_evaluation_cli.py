import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hancut.eval.cli import main, read_predictions
from hancut.eval.zones import Thresholds, zone_of

ROOT = Path(__file__).resolve().parents[2]
HEADER = 'image_id,item,y_true,score\n'
VALID = HEADER + 'a,Gun,0,0.123456789\nb,Gun,1,0.876543219\nc,Knife,0,0.2\nd,Knife,1,0.8\n'


def write_csv(tmp_path, content=VALID, name='validation.csv'):
    path = tmp_path / name
    path.write_text(content, encoding='utf-8-sig')
    return path


def test_cli_exports_full_precision_server_compatible_thresholds(tmp_path):
    path = write_csv(tmp_path)
    output = tmp_path / 'results'
    process = subprocess.run([sys.executable, str(ROOT / 'ml/scripts/evaluate.py'), str(path), '--output-dir', str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    table = json.loads((output / 'thresholds.json').read_text())
    assert table['Gun']['low'] == 0.876543219
    assert {'default', 'Gun', 'Knife'} == set(table)
    spec = importlib.util.spec_from_file_location('server_judgment_contract', ROOT / 'server/app/judgment.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loaded = module.load_thresholds(output / 'thresholds.json')
    for name, t in loaded.items():
        for score in (0, t.low, t.high, 1):
            assert module.zone_of(score, t) == zone_of(score, Thresholds(t.low, t.high))
    assert '독립 시험셋 미제공' in (output / 'report.md').read_text()
    assert len(json.loads((output / 'metrics.json').read_text())['calibration']['overall']) == 3


@pytest.mark.parametrize('content', [
    '', HEADER, 'image_id,item,y_true\na,x,1\n',
    HEADER + 'a,x,1,nan\n', HEADER + 'a,x,1,inf\n', HEADER + 'a,x,1,-0.1\n',
    HEADER + 'a,x,1,1.1\n', HEADER + 'a,x,2,0.5\n', HEADER + 'a,x,1,no\n',
    HEADER + ',x,1,0.5\n', HEADER + 'a,,1,0.5\n', HEADER + 'a,default,1,0.5\n',
    HEADER + 'a,x,1\n', HEADER + 'a,x,1,0.5,extra\n',
    HEADER + 'a,x,1,0.5\na,x,0,0.4\n',
    'image_id,item,y_true,score,score\na,x,1,0.5,0.5\n',
])
def test_rejects_bad_csv(tmp_path, content):
    with pytest.raises(ValueError):
        read_predictions(write_csv(tmp_path, content))


@pytest.mark.parametrize('content', [HEADER + 'a,x,1,0.5\n', HEADER + 'a,x,0,1\nb,x,1,1\n'])
def test_unreliable_thresholds_not_exported(tmp_path, content):
    path = write_csv(tmp_path, content)
    output = tmp_path / 'out'
    with pytest.raises(SystemExit) as exc:
        main([str(path), '--output-dir', str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_holdout_uses_frozen_thresholds_and_unknown_item_fallback(tmp_path):
    validation = write_csv(tmp_path)
    test = write_csv(tmp_path, HEADER + 'e,Gun,1,0.01\nf,새품목,0,0.99\n', 'test.csv')
    out = tmp_path / 'out'
    main([str(validation), '--output-dir', str(out), '--test-csv', str(test)])
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['test']['전체']['miss_rate'] == 1
    assert metrics['test']['전체']['false_alarm_rate'] == 1
    assert metrics['test']['Gun']['false_alarm_rate'] is None
    assert metrics['test_default_items'] == ['새품목']


def test_rejects_overlap_and_overwrite(tmp_path):
    path = write_csv(tmp_path)
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        main([str(path), '--output-dir', str(out), '--test-csv', str(path)])
    assert not out.exists()
    main([str(path), '--output-dir', str(out)])
    before = (out / 'thresholds.json').read_bytes()
    with pytest.raises(SystemExit):
        main([str(path), '--output-dir', str(out)])
    assert (out / 'thresholds.json').read_bytes() == before


def test_optimized_thresholds_match_brute_force():
    import random
    from hancut.eval.zones import fit_thresholds
    rng = random.Random(42)
    for _ in range(50):
        ys = [0, 1] + [rng.randrange(2) for _ in range(30)]
        scores = [rng.choice([0, 0.1, 0.3, 0.7, 1]) for _ in ys]
        cap, alarm = rng.choice([0, 0.03, 0.5, 1]), rng.choice([0, 0.05, 0.5, 1])
        candidates = sorted({0., *scores, 1. + 1e-9})
        low = max(c for c in candidates if sum(y == 1 and s < c for y, s in zip(ys, scores)) / sum(ys) <= cap)
        highs = [c for c in candidates if c >= low and sum(y == 0 and s >= c for y, s in zip(ys, scores)) / ys.count(0) <= alarm]
        assert fit_thresholds(ys, scores, cap, alarm) == Thresholds(min(low, 1), min(min(highs), 1))


def test_one_image_may_have_a_row_per_item(tmp_path):
    # 품목 단위 이진 정의라 인덱스는 사진 한 장당 품목 수만큼 행을 가진다.
    # 예전에는 image_id 만으로 중복을 막아서 두 번째 품목 행에서 멈췄다.
    rows = read_predictions(write_csv(tmp_path, HEADER + 'x,Gun,1,0.9\nx,Knife,0,0.1\nx,Pliers,0,0.2\n'))
    assert [(r['image_id'], r['item']) for r in rows] == [('x', 'Gun'), ('x', 'Knife'), ('x', 'Pliers')]


def test_same_image_and_item_twice_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="중복 \\(image_id, item\\)"):
        read_predictions(write_csv(tmp_path, HEADER + 'x,Gun,1,0.9\nx,Gun,0,0.1\n'))


def _separable(n_pos, n_neg, items=("Gun", "Knife")):
    rows = []
    for item in items:
        rows += [f"p{item}{i},{item},1,0.9" for i in range(n_pos)]
        rows += [f"n{item}{i},{item},0,0.1" for i in range(n_neg)]
    return HEADER + "\n".join(rows) + "\n"


def test_upper_bound_empties_auto_clear_when_the_sample_is_small(tmp_path):
    # 품목당 양성 100장이면 1% 를 95% 로 보장할 수 없다 → low = 0 → 자동 통과 없음
    path = write_csv(tmp_path, _separable(100, 100))
    main([str(path), "--output-dir", str(tmp_path / "upper"), "--bound", "upper"])
    table = json.loads((tmp_path / "upper" / "thresholds.json").read_text())
    assert table["Gun"]["low"] == 0.0
    metrics = json.loads((tmp_path / "upper" / "metrics.json").read_text())
    assert metrics["bound"] == "upper" and metrics["delta"] == 0.05
    assert "신뢰 상한" in (tmp_path / "upper" / "report.md").read_text()

    main([str(path), "--output-dir", str(tmp_path / "emp")])
    assert json.loads((tmp_path / "emp" / "thresholds.json").read_text())["Gun"]["low"] > 0.1


def test_point_for_matches_the_cli_test_section(tmp_path):
    from hancut.eval.cli import point_for
    cal = write_csv(tmp_path, _separable(50, 50), "cal.csv")
    tst = write_csv(tmp_path, HEADER + "t1,Gun,1,0.05\nt2,Gun,0,0.95\nt3,Knife,1,0.9\nt4,Knife,0,0.1\n", "tst.csv")
    main([str(cal), "--output-dir", str(tmp_path / "o"), "--test-csv", str(tst)])
    table = json.loads((tmp_path / "o" / "thresholds.json").read_text())
    metrics = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert point_for(read_predictions(tst), table) == metrics["test"]["전체"]
