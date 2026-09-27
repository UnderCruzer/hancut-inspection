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
    return JudgmentZone.values.firstWhere(
      (zone) => zone.wireName == value,
      orElse: () => JudgmentZone.review,
    );
  }

  bool get needsReview => this == JudgmentZone.review;

  Color get color => switch (this) {
        JudgmentZone.autoClear => const Color(0xFF2E8657),
        JudgmentZone.review => const Color(0xFFD98A1E),
        JudgmentZone.autoAlarm => const Color(0xFFD9432A),
      };
}

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
