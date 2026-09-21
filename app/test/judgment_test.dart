import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hancut/main.dart';
import 'package:hancut/src/judgment.dart';

void main() {
  group('JudgmentZone', () {
    test('서버 문자열을 그대로 해석한다', () {
      expect(JudgmentZone.parse('auto_compliant'), JudgmentZone.autoCompliant);
      expect(JudgmentZone.parse('review'), JudgmentZone.review);
      expect(JudgmentZone.parse('auto_noncompliant'), JudgmentZone.autoNoncompliant);
    });

    test('모르는 값은 사람에게 보낸다', () {
      expect(JudgmentZone.parse('something_new'), JudgmentZone.review);
      expect(JudgmentZone.parse('').needsReview, isTrue);
    });

    test('확인 필요 구간만 사람 판단이 필요하다', () {
      expect(JudgmentZone.autoCompliant.needsReview, isFalse);
      expect(JudgmentZone.autoNoncompliant.needsReview, isFalse);
      expect(JudgmentZone.review.needsReview, isTrue);
    });
  });

  group('Judgment.fromJson', () {
    test('판정 응답을 읽는다', () {
      final judgment = Judgment.fromJson({
        'facility': '소형소화기',
        'score': 0.93,
        'zone': 'auto_noncompliant',
        'reasons': ['부식'],
        'model_version': 'v0.1',
      });

      expect(judgment.facility, '소형소화기');
      expect(judgment.score, 0.93);
      expect(judgment.zone, JudgmentZone.autoNoncompliant);
      expect(judgment.reasons, ['부식']);
      expect(judgment.modelVersion, 'v0.1');
    });

    test('사유가 없어도 깨지지 않는다', () {
      final judgment = Judgment.fromJson({
        'facility': '방화문',
        'score': 0.1,
        'zone': 'auto_compliant',
      });

      expect(judgment.reasons, isEmpty);
      expect(judgment.modelVersion, 'unknown');
    });
  });

  testWidgets('홈 화면에 세 구간과 점검 시작 버튼이 보인다', (tester) async {
    await tester.pumpWidget(const HancutApp());

    expect(find.text('한컷점검'), findsOneWidget);
    expect(find.text('준수'), findsOneWidget);
    expect(find.text('확인 필요'), findsOneWidget);
    expect(find.text('미준수'), findsOneWidget);

    final button = tester.widget<FilledButton>(find.byType(FilledButton));
    expect(button.onPressed, isNull, reason: '촬영 화면은 W9에 붙인다');
  });
}
