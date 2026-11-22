import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hancut/src/api.dart';
import 'package:hancut/src/home_page.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:hancut/src/judgment.dart';

void main() {
  group('JudgmentZone', () {
    test('서버 문자열을 그대로 해석한다', () {
      expect(JudgmentZone.parse('auto_clear'), JudgmentZone.autoClear);
      expect(JudgmentZone.parse('review'), JudgmentZone.review);
      expect(JudgmentZone.parse('auto_alarm'), JudgmentZone.autoAlarm);
    });

    test('모르는 값은 사람에게 보낸다', () {
      expect(JudgmentZone.parse('something_new'), JudgmentZone.review);
      expect(JudgmentZone.parse('').needsReview, isTrue);
    });

    test('재검 구간만 사람 판단이 필요하다', () {
      expect(JudgmentZone.autoClear.needsReview, isFalse);
      expect(JudgmentZone.autoAlarm.needsReview, isFalse);
      expect(JudgmentZone.review.needsReview, isTrue);
    });
  });

  group('Judgment.fromJson', () {
    test('판정 응답을 읽는다', () {
      final judgment = Judgment.fromJson({
        'item': 'Gun',
        'score': 0.93,
        'zone': 'auto_alarm',
        'reasons': ['칼날 윤곽'],
        'model_version': 'v0.1',
      });

      expect(judgment.item, 'Gun');
      expect(judgment.score, 0.93);
      expect(judgment.zone, JudgmentZone.autoAlarm);
      expect(judgment.reasons, ['칼날 윤곽']);
      expect(judgment.modelVersion, 'v0.1');
    });

    test('사유가 없어도 깨지지 않는다', () {
      final judgment = Judgment.fromJson({'item': 'Knife', 'score': 0.1, 'zone': 'auto_clear'});

      expect(judgment.reasons, isEmpty);
      expect(judgment.modelVersion, 'unknown');
    });
  });

  group('Inspection', () {
    test('박스와 사진 크기를 읽고, 잘못된 박스는 버린다', () {
      final r = Inspection.fromJson({
        'items': [
          {'item': 'Knife', 'score': 0.5, 'zone': 'review', 'box': [1, 2, 30, 40]},
          {'item': 'Gun', 'score': 0.1, 'zone': 'auto_clear', 'box': [30, 2, 1, 40]},
          {'item': 'Lighter', 'score': 0.1, 'zone': 'auto_clear'},
        ],
        'image_size': [640, 480],
      });
      expect(r.imageSize, const Size(640, 480));
      expect(r.items[0].box, const Rect.fromLTRB(1, 2, 30, 40));
      expect(r.items[1].box, isNull);
      expect(r.items[2].box, isNull);
    });

    Map<String, dynamic> item(String name, double score, String zone) => {
      'item': name,
      'score': score,
      'zone': zone,
      'needs_review': zone == 'review',
      'thresholds': {'low': 0.1, 'high': 0.9},
    };

    test('품목 하나라도 적발이면 사진도 적발', () {
      final r = Inspection.fromJson({
        'items': [item('Knife', 0.05, 'auto_clear'), item('Gun', 0.95, 'auto_alarm'), item('Lighter', 0.4, 'review')],
        'model_version': 'e1',
      });
      expect(r.overall, JudgmentZone.autoAlarm);
      expect(r.sorted.map((e) => e.item), ['Gun', 'Lighter', 'Knife']);
      expect(r.inZone(JudgmentZone.review).single.label, '라이터');
    });

    test('재검이 있으면 재검, 모두 통과일 때만 통과', () {
      expect(
        Inspection.fromJson({
          'items': [item('Knife', 0.05, 'auto_clear'), item('Lighter', 0.4, 'review')],
        }).overall,
        JudgmentZone.review,
      );
      expect(
        Inspection.fromJson({
          'items': [item('Knife', 0.05, 'auto_clear')],
        }).overall,
        JudgmentZone.autoClear,
      );
    });

    test('실험 표시가 없으면 실험용으로 본다', () {
      expect(
        Inspection.fromJson({
          'items': [item('Knife', 0.05, 'auto_clear')],
        }).experimental,
        isTrue,
      );
    });

    test('빈 판정은 받지 않는다', () {
      expect(() => Inspection.fromJson({'items': []}), throwsFormatException);
    });
  });

  group('InspectionClient', () {
    final png = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0]);
    final jpeg = Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0, 0, 0]);

    test('확장자 대신 내용으로 형식을 정한다', () {
      expect(sniffImageType(png), 'png');
      expect(sniffImageType(jpeg), 'jpeg');
      expect(sniffImageType(Uint8List.fromList([1, 2, 3])), isNull);
    });

    test('v2 판정을 보내고 읽는다', () async {
      late http.BaseRequest sent;
      final client = InspectionClient(
        'http://server/',
        client: MockClient((request) async {
          sent = request;
          return http.Response.bytes(
            utf8.encode(
              jsonEncode({
                'items': [
                  {
                    'item': 'Gun',
                    'score': 0.97,
                    'zone': 'auto_alarm',
                    'thresholds': {'low': 0.01, 'high': 0.6},
                  },
                ],
                'model_version': 'e1',
                'experimental': true,
              }),
            ),
            200,
            headers: {'content-type': 'application/json'},
          );
        }),
      );
      final r = await client.inspect(jpeg);
      expect(sent.url.toString(), 'http://server/v2/inspections');
      expect(r.overall, JudgmentZone.autoAlarm);
      expect(r.items.single.label, '총기');
    });

    test('서버 오류를 한국어 문장으로 바꾼다', () async {
      final client = InspectionClient('http://server', client: MockClient((_) async => http.Response('{}', 503)));
      await expectLater(
        client.inspect(png),
        throwsA(isA<InspectionError>().having((e) => e.message, 'message', contains('준비'))),
      );
    });

    test('사진이 아니면 보내지 않는다', () async {
      var called = false;
      final client = InspectionClient(
        'http://server',
        client: MockClient((_) async {
          called = true;
          return http.Response('{}', 200);
        }),
      );
      await expectLater(client.inspect(Uint8List.fromList([1, 2, 3])), throwsA(isA<InspectionError>()));
      expect(called, isFalse);
    });
  });

  test('막대는 로그 눈금, 작은 점수는 지수로 쓴다', () {
    expect(scorePosition(1), 1);
    expect(scorePosition(1e-6), closeTo(0, 1e-9));
    expect(scorePosition(1e-3), closeTo(0.5, 1e-9));
    expect(scorePosition(0), 0);
    expect(formatScore(0.4071), '0.407');
    expect(formatScore(0.00004), '4e-5');
    expect(shortVersion('sha256:2d41d3d38914c27f5275'), 'sha256:2d41d3d38914');
  });

  testWidgets('사진을 고르면 사진 판정과 품목별 구간을 보여준다', (tester) async {
    tester.view.physicalSize = const Size(1400, 1000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final image = Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0, 0, 0]);
    await tester.pumpWidget(
      MaterialApp(
        home: HomePage(
          pickImage: () async => (image, 'bag.jpg'),
          inspector: InspectionClient(
            'http://server',
            client: MockClient(
              (_) async => http.Response.bytes(
                utf8.encode(
                  jsonEncode({
                    'items': [
                      {
                        'item': 'Lighter',
                        'score': 0.42,
                        'zone': 'review',
                        'thresholds': {'low': 0.1, 'high': 0.9},
                        'box': [10, 20, 60, 80],
                      },
                      {
                        'item': 'Knife',
                        'score': 0.01,
                        'zone': 'auto_clear',
                        'thresholds': {'low': 0.1, 'high': 0.9},
                        'box': [100, 100, 150, 140],
                      },
                    ],
                    'image_size': [200, 160],
                    'model_version': 'e1',
                  }),
                ),
                200,
              ),
            ),
          ),
        ),
      ),
    );
    expect(find.text('재검'), findsOneWidget, reason: '처음에는 구간 안내만 보인다');

    await tester.tap(find.text('X-ray 사진 올려서 판정하기'));
    await tester.pumpAndSettle();

    expect(tester.widget<Text>(find.byKey(const Key('overall'))).data, '재검');
    expect(find.text('판독관 확인 필요: 라이터'), findsOneWidget);
    expect(find.text('칼'), findsOneWidget);
    expect(find.byKey(const Key('box')), findsOneWidget, reason: '처음에는 가장 엄한 품목(라이터) 자리를 표시한다');
    expect(find.textContaining('라이터 — 이 사진에서'), findsOneWidget);

    await tester.tap(find.byKey(const Key('item-Knife')));
    await tester.pumpAndSettle();
    expect(find.textContaining('칼 — 이 사진에서'), findsOneWidget);
    expect(find.textContaining('물건이 아닐 가능성'), findsOneWidget, reason: '통과 품목 자리는 물건이 아닐 수 있다고 알린다');
  });
}
