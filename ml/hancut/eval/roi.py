"""Pure geometry and calibration gates for the ROI refinement experiment."""
import math


def crop_window(box, width, height, context=0.2):
    x, y, r, b = map(float, box)
    if not all(math.isfinite(v) for v in (x,y,r,b)) or r <= x or b <= y or width < 1 or height < 1 or context < 0:
        raise ValueError("invalid crop")
    dx, dy = (r-x)*context, (b-y)*context
    out = (max(0, math.floor(x-dx)), max(0, math.floor(y-dy)),
           min(width, math.ceil(r+dx)), min(height, math.ceil(b+dy)))
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError("crop outside image")
    return out


def covered_fraction(window, gt):
    """Fraction of a GT object covered, avoiding 'background' crops containing tiny objects."""
    x,y,r,b = window
    a,c,d,e = gt
    area = (d-a)*(e-c)
    if area <= 0:
        raise ValueError("invalid GT")
    return max(0,min(r,d)-max(x,a))*max(0,min(b,e)-max(y,c))/area


def proposal_scores(old_class, old_score, probs, targets, alphas):
    if len(probs) < max(targets)+1 or not 0 <= old_score <= 1 or any(not 0 <= p <= 1 for p in probs):
        raise ValueError("invalid probabilities")
    scores = {}
    for target in targets:
        alpha = alphas[target]
        if not 0 <= alpha <= 1:
            raise ValueError("invalid alpha")
        baseline = old_score if old_class == target else 0.0
        scores[target] = (1-alpha)*baseline + alpha*old_score*probs[target]
    return scores


def choose_alpha(baseline, candidates, gate):
    """Choose only from calib; alpha=0 is the explicit no-change outcome."""
    valid = [(0.0,baseline)]
    for alpha, point in candidates:
        if (alpha and point['ap50'] >= baseline['ap50'] + gate['ap50_gain']
            and point['ap50_95'] >= baseline['ap50_95'] - gate['ap50_95_drop']
            and point['precision'] is not None and baseline['precision'] is not None
            and point['precision'] >= baseline['precision'] - gate['precision_drop']
            and point['recall'] >= baseline['recall'] - gate['recall_drop']):
            valid.append((alpha,point))
    return max(valid,key=lambda pair:(pair[1]['ap50'],-pair[0]))[0]


def assert_disjoint(split_ids):
    seen = set()
    for split, ids in split_ids.items():
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"duplicate images or split leakage: {split}")
        seen.update(ids)
