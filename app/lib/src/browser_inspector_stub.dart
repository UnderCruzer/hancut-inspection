import 'dart:typed_data';

import 'api.dart';
import 'judgment.dart';

const browserInspectionAvailable = false;

class BrowserInspector implements Inspector {
  @override
  String get description => '이 기기에서는 브라우저 판정을 쓸 수 없어요';

  @override
  Future<void> warmup() async => throw const InspectionError('브라우저 판정은 웹에서만 쓸 수 있어요.');

  @override
  Future<Inspection> inspect(Uint8List bytes, {String filename = 'xray'}) async =>
      throw const InspectionError('브라우저 판정은 웹에서만 쓸 수 있어요.');
}
