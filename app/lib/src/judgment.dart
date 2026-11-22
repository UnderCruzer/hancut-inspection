import 'package:flutter/material.dart';

/// 판정 3구간 — 서버(`server/app/judgment.py`), 학습(`ml/hancut/eval/zones.py`)과
/// 같은 문자열 규약을 쓴다. 값이 어긋나면 앱이 잘못된 화면을 보여준다.
enum JudgmentZone {
  autoClear('auto_clear', '통과', '판독관 확인 없이 통과합니다'),
  review('review', '재검', '판독관이 직접 확인해야 합니다'),
  autoAlarm('auto_alarm', '적발', '개봉 검사 대상으로 표시됩니다');

  const JudgmentZone(this.wireName, this.label, this.description);

  final String wireName;
  final String label;
  final String description;

  /// 모르는 값이 오면 조용히 넘기지 않고 '재검'으로 보낸다 — 사람이 보게 한다.
  static JudgmentZone parse(String value) {
    return JudgmentZone.values.firstWhere((zone) => zone.wireName == value, orElse: () => JudgmentZone.review);
  }

  bool get needsReview => this == JudgmentZone.review;

  /// 사진 한 장의 판정에서 무엇을 먼저 보여줄지. 적발 > 재검 > 통과.
  int get severity => switch (this) {
    JudgmentZone.autoClear => 0,
    JudgmentZone.review => 1,
    JudgmentZone.autoAlarm => 2,
  };

  Color get color => switch (this) {
    JudgmentZone.autoClear => const Color(0xFF2E8657),
    JudgmentZone.review => const Color(0xFFD98A1E),
    JudgmentZone.autoAlarm => const Color(0xFFD9432A),
  };
}

/// 서버 품목 이름(PIDray 클래스) → 화면에 보일 이름.
const itemLabels = {
  'Baton': '경봉',
  'Pliers': '펜치',
  'Hammer': '망치',
  'Powerbank': '보조배터리',
  'Scissors': '가위',
  'Wrench': '렌치',
  'Gun': '총기',
  'Bullet': '탄약',
  'Sprayer': '스프레이',
  'HandCuffs': '수갑',
  'Knife': '칼',
  'Lighter': '라이터',
};

String itemLabel(String item) => itemLabels[item] ?? item;

/// 서버 `POST /v1/inspections` 응답.
class Judgment {
  const Judgment({
    required this.item,
    required this.score,
    required this.zone,
    required this.reasons,
    required this.modelVersion,
  });

  final String item;
  final double score;
  final JudgmentZone zone;
  final List<String> reasons;
  final String modelVersion;

  factory Judgment.fromJson(Map<String, dynamic> json) {
    return Judgment(
      item: json['item'] as String,
      score: (json['score'] as num).toDouble(),
      zone: JudgmentZone.parse(json['zone'] as String),
      reasons: (json['reasons'] as List<dynamic>? ?? const []).map((e) => e.toString()).toList(),
      modelVersion: json['model_version'] as String? ?? 'unknown',
    );
  }
}

/// `POST /v2/inspections` 의 품목 하나.
class ItemJudgment {
  const ItemJudgment({
    required this.item,
    required this.score,
    required this.zone,
    required this.low,
    required this.high,
    this.box,
  });

  final String item;
  final double score;
  final JudgmentZone zone;
  final double low;
  final double high;

  /// 그 품목 점수가 가장 높은 자리 (원본 사진 좌표 x1, y1, x2, y2). 검출 결과가 아니라 '가장 의심되는 곳'이다.
  final Rect? box;

  String get label => itemLabel(item);

  factory ItemJudgment.fromJson(Map<String, dynamic> json) {
    final thresholds = json['thresholds'] as Map<String, dynamic>? ?? const {};
    return ItemJudgment(
      item: json['item'] as String,
      score: (json['score'] as num).toDouble(),
      zone: JudgmentZone.parse(json['zone'] as String),
      low: (thresholds['low'] as num?)?.toDouble() ?? double.nan,
      high: (thresholds['high'] as num?)?.toDouble() ?? double.nan,
      box: _rect(json['box']),
    );
  }
}

Rect? _rect(Object? raw) {
  if (raw is! List || raw.length != 4) return null;
  final v = raw.map((e) => (e as num).toDouble()).toList();
  return v[2] > v[0] && v[3] > v[1] ? Rect.fromLTRB(v[0], v[1], v[2], v[3]) : null;
}

/// 사진 한 장의 판정 — 12종 품목별 구간과, 그중 가장 엄한 구간.
class Inspection {
  const Inspection({required this.items, required this.modelVersion, required this.experimental, this.imageSize});

  final List<ItemJudgment> items;

  /// 박스 좌표의 기준이 되는 원본 사진 크기.
  final Size? imageSize;
  final String modelVersion;
  final bool experimental;

  factory Inspection.fromJson(Map<String, dynamic> json) {
    final items = (json['items'] as List<dynamic>? ?? const [])
        .map((e) => ItemJudgment.fromJson(e as Map<String, dynamic>))
        .toList();
    if (items.isEmpty) {
      throw const FormatException('품목 판정이 비어 있다');
    }
    return Inspection(
      items: items,
      modelVersion: json['model_version'] as String? ?? 'unknown',
      // 표시가 없으면 실험용으로 본다 — 보장된 결과처럼 보이지 않게 한다.
      experimental: json['experimental'] as bool? ?? true,
      imageSize: switch (json['image_size']) {
        [num w, num h] when w > 0 && h > 0 => Size(w.toDouble(), h.toDouble()),
        _ => null,
      },
    );
  }

  /// 품목 하나라도 적발이면 적발, 재검이 있으면 재검, 모두 통과일 때만 통과.
  JudgmentZone get overall => items.map((e) => e.zone).reduce((a, b) => a.severity >= b.severity ? a : b);

  /// 엄한 구간 먼저, 같은 구간에서는 점수가 높은 품목 먼저.
  List<ItemJudgment> get sorted => [...items]
    ..sort((a, b) {
      final bySeverity = b.zone.severity.compareTo(a.zone.severity);
      return bySeverity != 0 ? bySeverity : b.score.compareTo(a.score);
    });

  List<ItemJudgment> inZone(JudgmentZone zone) => sorted.where((e) => e.zone == zone).toList();
}
