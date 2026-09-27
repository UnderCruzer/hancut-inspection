import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hancut.eval.cli import main, read_predictions
from hancut.eval.zones import Thresholds, zone_of

ROOT = Path(__file__).resolve().parents[2]
HEADER = 'image_id,facility,y_true,score\n'
VALID = HEADER + 'a,소화기,0,0.123456789\nb,소화기,1,0.876543219\nc,방화문,0,0.2\nd,방화문,1,0.8\n'


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
    assert table['소화기']['low'] == 0.876543219
    assert {'default', '소화기', '방화문'} == set(table)
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
    '', HEADER, 'image_id,facility,y_true\na,x,1\n',
    HEADER + 'a,x,1,nan\n', HEADER + 'a,x,1,inf\n', HEADER + 'a,x,1,-0.1\n',
    HEADER + 'a,x,1,1.1\n', HEADER + 'a,x,2,0.5\n', HEADER + 'a,x,1,no\n',
    HEADER + ',x,1,0.5\n', HEADER + 'a,,1,0.5\n', HEADER + 'a,default,1,0.5\n',
    HEADER + 'a,x,1\n', HEADER + 'a,x,1,0.5,extra\n',
    HEADER + 'a,x,1,0.5\na,y,0,0.4\n',
    'image_id,facility,y_true,score,score\na,x,1,0.5,0.5\n',
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


def test_holdout_uses_frozen_thresholds_and_unknown_facility_fallback(tmp_path):
    validation = write_csv(tmp_path)
    test = write_csv(tmp_path, HEADER + 'e,소화기,1,0.01\nf,새시설,0,0.99\n', 'test.csv')
    out = tmp_path / 'out'
    main([str(validation), '--output-dir', str(out), '--test-csv', str(test)])
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['test']['전체']['miss_rate'] == 1
    assert metrics['test']['전체']['false_alarm_rate'] == 1
    assert metrics['test']['소화기']['false_alarm_rate'] is None
    assert metrics['test_default_facilities'] == ['새시설']


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
